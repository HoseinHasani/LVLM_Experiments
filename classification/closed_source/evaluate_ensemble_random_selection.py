import os
import json
import glob
import pickle
import random

from chair import CHAIR


# ====================
# Config
# ====================
ENS_FOLDER = "ens_data"
COCO_PATH = "coco_annotations"
CACHE_PATH = "chair.pkl"

RANDOM_SEED = 22

OUT_DIR = "chair_ensemble/random_selection"
os.makedirs(OUT_DIR, exist_ok=True)

TMP_RANDOM_CAPS = os.path.join(
    OUT_DIR, f"random_single_captions_seed_{RANDOM_SEED}.jsonl"
)
SAVE_DATASET_METRICS = os.path.join(
    OUT_DIR, f"chair_random_dataset_metrics_seed_{RANDOM_SEED}.json"
)
SAVE_PER_IMAGE_METRICS = os.path.join(
    OUT_DIR, f"chair_random_per_image_metrics_seed_{RANDOM_SEED}.json"
)

IMAGE_ID_KEY = "image_id"
CAPTION_KEY = "caption"


# ====================
# Load / build CHAIR evaluator
# ====================
if os.path.exists(CACHE_PATH):
    evaluator = pickle.load(open(CACHE_PATH, "rb"))
    print(f"Loaded CHAIR evaluator from cache: {CACHE_PATH}")
else:
    evaluator = CHAIR(COCO_PATH)
    pickle.dump(evaluator, open(CACHE_PATH, "wb"))
    print(f"Built and cached CHAIR evaluator: {CACHE_PATH}")


random.seed(RANDOM_SEED)

jsonl_files = sorted(glob.glob(os.path.join(ENS_FOLDER, "*.jsonl")))
assert len(jsonl_files) > 0, "No jsonl files found in ens_data/"


# ====================
# 1) Randomly select ONE caption per image
# ====================
selected_entries = []

for path in jsonl_files:
    with open(path, "r") as f:
        entries = [json.loads(l) for l in f]

    assert len(entries) > 0, f"No captions in {path}"
    chosen = random.choice(entries)
    selected_entries.append(chosen)

with open(TMP_RANDOM_CAPS, "w") as f:
    for entry in selected_entries:
        f.write(json.dumps(entry) + "\n")

print(f"Selected {len(selected_entries)} random captions")
print(f"→ {TMP_RANDOM_CAPS}")


# ====================
# 2) Dataset-level CHAIR
# ====================
dataset_output = evaluator.compute_chair(
    cap_file=TMP_RANDOM_CAPS,
    image_id_key=IMAGE_ID_KEY,
    caption_key=CAPTION_KEY,
)

dataset_metrics = dataset_output["overall_metrics"]

with open(SAVE_DATASET_METRICS, "w") as f:
    json.dump(dataset_metrics, f, indent=2)

print("\n=== DATASET-LEVEL CHAIR (RANDOM SELECTION) ===")
print(f"Random seed: {RANDOM_SEED}")
for k, v in dataset_metrics.items():
    print(f"{k:10s}: {v * 100:.2f}")


# ====================
# 3) Per-image metrics
# ====================
per_image_metrics = {}

for sent in dataset_output["sentences"]:
    per_image_metrics[sent["image_id"]] = {
        "image_id": sent["image_id"],
        "gen_id": sent.get("gen_id"),
        "metrics": sent["metrics"],
        "caption": sent["caption"],
    }

with open(SAVE_PER_IMAGE_METRICS, "w") as f:
    json.dump(per_image_metrics, f, indent=2)

print(f"\nSaved per-image metrics to:")
print(f"→ {SAVE_PER_IMAGE_METRICS}")


# ====================
# Cleanup
# ====================
os.remove(TMP_RANDOM_CAPS)
