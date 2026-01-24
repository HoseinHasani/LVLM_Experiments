import os
import json
import glob
import pickle
from collections import defaultdict

from chair import CHAIR


# ====================
# Config
# ====================
ENS_FOLDER = "ens_data"
COCO_PATH = "coco_annotations"
CACHE_PATH = "chair.pkl"

OUT_DIR = "chair_ensemble/raw_averaged"
os.makedirs(OUT_DIR, exist_ok=True)

TMP_ALL_CAPS = os.path.join(OUT_DIR, "all_ensemble_captions.jsonl")
SAVE_DATASET_METRICS = os.path.join(OUT_DIR, "chair_raw_ensemble_dataset_metrics.json")
SAVE_PER_IMAGE_METRICS = os.path.join(OUT_DIR, "chair_raw_ensemble_per_image_metrics.json")

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


# ====================
# Collect ensemble files
# ====================
jsonl_files = sorted(glob.glob(os.path.join(ENS_FOLDER, "*.jsonl")))
assert len(jsonl_files) > 0, "No jsonl files found in ens_data/"


# ====================
# 1) Concatenate ALL ensemble captions
# ====================
with open(TMP_ALL_CAPS, "w") as out:
    for path in jsonl_files:
        with open(path, "r") as f:
            for line in f:
                out.write(line)

print(f"Concatenated {len(jsonl_files)} files")
print(f"→ {TMP_ALL_CAPS}")


# ====================
# 2) Dataset-level CHAIR (raw ensemble)
# ====================
dataset_output = evaluator.compute_chair(
    cap_file=TMP_ALL_CAPS,
    image_id_key=IMAGE_ID_KEY,
    caption_key=CAPTION_KEY,
)

dataset_metrics = dataset_output["overall_metrics"]

with open(SAVE_DATASET_METRICS, "w") as f:
    json.dump(dataset_metrics, f, indent=2)

print("\n=== DATASET-LEVEL CHAIR (RAW ENSEMBLE) ===")
for k, v in dataset_metrics.items():
    print(f"{k:10s}: {v * 100:.2f}")


# ====================
# 3) Per-image ensemble metrics
# ====================
per_image_results = {}

for path in jsonl_files:
    image_name = os.path.basename(path).replace(".jsonl", "")

    cap_dict = evaluator.compute_chair(
        cap_file=path,
        image_id_key=IMAGE_ID_KEY,
        caption_key=CAPTION_KEY,
    )

    sentences = cap_dict["sentences"]
    num_caps = len(sentences)

    metric_sum = defaultdict(float)

    for sent in sentences:
        for k, v in sent["metrics"].items():
            if k in ["F1", "Len"]:
                continue
            metric_sum[k] += v

    avg_metrics = {k: metric_sum[k] / num_caps for k in metric_sum}

    per_image_results[image_name] = {
        "image_id": sentences[0]["image_id"],
        "ensemble_size": num_caps,
        "avg_caption_metrics": avg_metrics,
        "micro_ensemble_metrics": cap_dict["overall_metrics"],
    }

with open(SAVE_PER_IMAGE_METRICS, "w") as f:
    json.dump(per_image_results, f, indent=2)

print(f"\nSaved per-image ensemble metrics to:")
print(f"→ {SAVE_PER_IMAGE_METRICS}")


# ====================
# Cleanup
# ====================
os.remove(TMP_ALL_CAPS)
