import os
import json
import glob
from collections import defaultdict
import pickle
from chair import CHAIR  

ENS_FOLDER = "ens_data"
COCO_PATH = "coco_annotations"
CACHE_PATH = "chair.pkl"
SAVE_PATH = "chair_ensemble_results.json"

IMAGE_ID_KEY = "image_id"
CAPTION_KEY = "caption"


if os.path.exists(CACHE_PATH):
    evaluator = pickle.load(open(CACHE_PATH, "rb"))
else:
    evaluator = CHAIR(COCO_PATH)
    pickle.dump(evaluator, open(CACHE_PATH, "wb"))

results = {}

jsonl_files = sorted(glob.glob(os.path.join(ENS_FOLDER, "*.jsonl")))

for path in jsonl_files:
    image_name = os.path.basename(path).replace(".jsonl", "")

    cap_dict = evaluator.compute_chair(
        cap_file=path,
        image_id_key=IMAGE_ID_KEY,
        caption_key=CAPTION_KEY,
    )

    # --- average metrics over ensemble captions ---
    metrics_sum = defaultdict(float)
    num_caps = len(cap_dict["sentences"])

    for sent in cap_dict["sentences"]:
        for k, v in sent["metrics"].items():
            metrics_sum[k] += v

    avg_metrics = {k: metrics_sum[k] / num_caps for k in metrics_sum}

    results[image_name] = {
        "image_id": cap_dict["sentences"][0]["image_id"],
        "ensemble_size": num_caps,
        "avg_metrics": avg_metrics,
        "overall_metrics_raw": cap_dict["overall_metrics"],  # optional
    }

# Save results
with open(SAVE_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved ensemble-averaged CHAIR results to {SAVE_PATH}")


