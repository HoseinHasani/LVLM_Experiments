import os
import json
import glob
import pickle
from collections import defaultdict
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from chair import CHAIR


# ====================
# Config
# ====================

THRESHOLD_RATIO = 0.7
FILTER_COCO_OBJ = True

ENS_FOLDER = "ens_data"
COCO_PATH = "coco_annotations"
CACHE_PATH = "chair.pkl"
COCO_OBJ_PATH = "mscoco_objects.npy"

OUT_DIR = f"chair_ensemble/smart_selection_filter_{FILTER_COCO_OBJ}"
os.makedirs(OUT_DIR, exist_ok=True)

TMP_SELECTED_CAPS = os.path.join(
    OUT_DIR, f"selected_captions_thr_{THRESHOLD_RATIO:.2f}.jsonl"
)
SAVE_DATASET_METRICS = os.path.join(
    OUT_DIR, f"chair_dataset_metrics_thr_{THRESHOLD_RATIO:.2f}.json"
)
SAVE_PER_IMAGE_INFO = os.path.join(
    OUT_DIR, f"selection_details_thr_{THRESHOLD_RATIO:.2f}.json"
)

IMAGE_ID_KEY = "image_id"
CAPTION_KEY = "caption"

if FILTER_COCO_OBJ:
    mscoco_objects = np.load(COCO_OBJ_PATH)
else:
    mscoco_objects = None

# ====================
# NLTK setup (download once if needed)
# ====================
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("wordnet")


# ====================
# CHAIR evaluator
# ====================
if os.path.exists(CACHE_PATH):
    evaluator = pickle.load(open(CACHE_PATH, "rb"))
    print(f"Loaded CHAIR evaluator from cache: {CACHE_PATH}")
else:
    evaluator = CHAIR(COCO_PATH)
    pickle.dump(evaluator, open(CACHE_PATH, "wb"))
    print(f"Built and cached CHAIR evaluator: {CACHE_PATH}")


# ====================
# NLP utilities
# ====================
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def process_caption(caption, mscoco_objects):
    words = nltk.word_tokenize(caption.lower())
    tagged = nltk.pos_tag(words)

    wnl = WordNetLemmatizer()

    tokens = []
    lemma_to_indices = defaultdict(list)

    for idx, (word, tag) in enumerate(tagged):
        if tag.startswith("NN"):
            lemma = wnl.lemmatize(word, pos=wordnet.NOUN)
            is_noun = True
        else:
            lemma = word
            is_noun = False
            
        is_noun = tag in {"NN", "NNS"} # Exclude NNP / NNPS

        token = {
            "word": word,
            "lemma": lemma,
            "pos": tag,
            "is_noun": is_noun,
            "position": idx,
            "is_first_occurrence": False,
            "group_indices": None,
            "is_coco_obj": None
        }
        
        if mscoco_objects is not None:
            is_coco_obj = word in mscoco_objects or lemma in mscoco_objects
            token["is_coco_obj"] = is_coco_obj
            
        tokens.append(token)

        if is_noun:
            lemma_to_indices[lemma].append(idx)

    for lemma, indices in lemma_to_indices.items():
        first_idx = indices[0]
        for idx in indices:
            tokens[idx]["is_first_occurrence"] = (idx == first_idx)
            tokens[idx]["group_indices"] = indices

    return tokens


def score_caption(caption, threshold_pos, mscoco_objects):
    tokens = process_caption(caption, mscoco_objects)

    real_num = 0
    hall_num = 0
    hall_tot_repeat = 0

    for tok in tokens:
        if not tok["is_noun"]:
            continue
        if not tok["is_first_occurrence"]:
            continue
        if tok["is_coco_obj"] is not None:
            if not tok["is_coco_obj"]:
                continue

        if tok["position"] < threshold_pos:
            real_num += 1
        else:
            hall_num += 1
            hall_tot_repeat += len(tok["group_indices"])

    print(real_num, hall_num, hall_tot_repeat)
    score = 0.4 * real_num - hall_num - 0.4 * hall_tot_repeat

    return {
        "score": score,
        "real_num": real_num,
        "hall_num": hall_num,
        "hall_tot_repeat": hall_tot_repeat,
    }


def select_best_caption(captions, threshold_ratio, mscoco_objects):
    lengths = [len(process_caption(c, mscoco_objects)) for c in captions]
    avg_len = sum(lengths) / len(lengths)
    threshold_pos = int(threshold_ratio * avg_len)

    best_idx = None
    best_score = float("-inf")
    all_scores = []

    for i, cap in enumerate(captions):
        res = score_caption(cap, threshold_pos, mscoco_objects)
        all_scores.append(res)

        if res["score"] > best_score:
            best_score = res["score"]
            best_idx = i

    return {
        "best_index": best_idx,
        "best_score": best_score,
        "avg_length": avg_len,
        "threshold_pos": threshold_pos,
        "all_scores": all_scores,
    }


# ====================
# Main loop
# ====================
jsonl_files = sorted(glob.glob(os.path.join(ENS_FOLDER, "*.jsonl")))
assert len(jsonl_files) > 0, "No jsonl files found in ens_data/"

selected_entries = []
selection_details = {}

for path in jsonl_files:
    with open(path, "r") as f:
        entries = [json.loads(l) for l in f]

    captions = [e["caption"] for e in entries]

    sel = select_best_caption(captions, THRESHOLD_RATIO, mscoco_objects)
    chosen_entry = entries[sel["best_index"]]

    selected_entries.append(chosen_entry)

    image_key = os.path.basename(path).replace(".jsonl", "")
    selection_details[image_key] = {
        "image_id": chosen_entry["image_id"],
        "chosen_gen_id": chosen_entry.get("gen_id"),
        "selection_info": sel,
    }


# ====================
# Save selected captions
# ====================
with open(TMP_SELECTED_CAPS, "w") as f:
    for e in selected_entries:
        f.write(json.dumps(e) + "\n")

print(f"Saved selected captions to {TMP_SELECTED_CAPS}")


# ====================
# Run CHAIR
# ====================
dataset_output = evaluator.compute_chair(
    cap_file=TMP_SELECTED_CAPS,
    image_id_key=IMAGE_ID_KEY,
    caption_key=CAPTION_KEY,
)

dataset_metrics = dataset_output["overall_metrics"]

with open(SAVE_DATASET_METRICS, "w") as f:
    json.dump(dataset_metrics, f, indent=2)

with open(SAVE_PER_IMAGE_INFO, "w") as f:
    json.dump(selection_details, f, indent=2)


print("\n=== CHAIR RESULTS (SMART SELECTION) ===")
print(f"Threshold ratio: {THRESHOLD_RATIO}")
for k, v in dataset_metrics.items():
    print(f"{k:10s}: {v * 100:.2f}")

print(f"\nResults saved in folder: {OUT_DIR}")
