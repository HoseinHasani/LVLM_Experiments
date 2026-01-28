import spacy
from collections import Counter, defaultdict

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def extract_word_metadata(
    caption: str,
    hallucinated_words: list,
    non_hallucinated_words: list
):
    doc = nlp(caption)

    hallucinated_lemmas = {
        t.lemma_.lower()
        for w in hallucinated_words
        for t in nlp(w)
        if not t.is_space and not t.is_punct
    }

    non_hallucinated_lemmas = {
        t.lemma_.lower()
        for w in non_hallucinated_words
        for t in nlp(w)
        if not t.is_space and not t.is_punct
    }

    words = []
    lemmas = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue
        words.append(token.text)
        lemmas.append(token.lemma_.lower())

    lemma_total_counter = Counter(lemmas)
    lemma_occ_counter = defaultdict(int)


    lemma_positions = defaultdict(list)
    for idx, lemma in enumerate(lemmas):
        lemma_positions[lemma].append(idx)


    word_metadata = []

    for idx, (word, lemma) in enumerate(zip(words, lemmas)):
        lemma_occ_counter[lemma] += 1
        is_first = lemma_occ_counter[lemma] == 1

        if lemma in hallucinated_lemmas:
            wtype = "fp"
        elif lemma in non_hallucinated_lemmas:
            wtype = "tp"
        else:
            wtype = "other"

        word_metadata.append({
            "word": word,
            "lemma": lemma,
            "position": idx,              # WORD INDEX
            "rep_num": lemma_occ_counter[lemma],
            "total_reps": lemma_total_counter[lemma],
            "is_first_occ": is_first,
            "all_occurrence_indices": lemma_positions[lemma] if is_first else None,
            "type": wtype
        })

    return word_metadata


import json
import os
from tqdm import tqdm

def load_entries_from_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    # Case B: {"sentences": [...]}
    if isinstance(data, dict) and "sentences" in data:
        return data["sentences"]

    # Case A: single entry
    if isinstance(data, dict) and "image_id" in data:
        return [data]

    raise ValueError(f"Unrecognized format in {path}")


def eval_minigpt_metadata_only_from_dir(
    responses_dir,
    output_dir,
    chair_results
):
    """
    responses_dir: directory containing response JSON files
    output_dir: where per-image metadata JSONs are saved
    chair_results: dict keyed by image_id (string or int)
    """

    os.makedirs(output_dir, exist_ok=True)

    response_files = sorted([
        os.path.join(responses_dir, f)
        for f in os.listdir(responses_dir)
        if f.endswith(".jsonl")
    ])

    for resp_file in tqdm(response_files, desc="Processing response files"):
        try:
            entries = load_entries_from_file(resp_file)
        except Exception as e:
            print(f"⚠️ Skipping {resp_file}: {e}")
            continue

        for entry in entries:
            image_id = entry["image_id"]
            caption = entry.get("caption", "").strip()

            if not caption:
                print(f"⚠️ Empty caption for image {image_id}")
                continue

            # --- Merge hallucinated / non-hallucinated from CHAIR ---
            chair_entry = chair_results.get(image_id) or {}

            hallucinated_words = [
                w[0] if isinstance(w, list) else w
                for w in chair_entry.get("mscoco_hallucinated_words", [])
            ]

            non_hallucinated_words = [
                w[0] if isinstance(w, list) else w
                for w in chair_entry.get("mscoco_true_positive_words", [])
            ]

            word_meta = extract_word_metadata(
                caption,
                hallucinated_words,
                non_hallucinated_words
            )

            out = {
                "image_id": image_id,
                "caption": caption,
                "words_meta": word_meta,
                "chair": chair_entry
            }

            out_path = os.path.join(output_dir, f"image_id_{image_id}.json")
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
 


if __name__ == "__main__":
    RESPONSES_DIR = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/Ensemble/experiments/new_per_sample_outputs/gpt4o_responses_greedy_jsonl"
    CHAIR_FILE   = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/Ensemble/experiments/new_per_sample_outputs/gpt4o_chairs_greedy.json"
    OUTPUT_DIR   = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/Ensemble/experiments/new_per_sample_outputs/gpt4o_meta_data"

    with open(CHAIR_FILE, "r") as f:
        chair_raw = json.load(f)

    chair_results = {
    entry["image_id"]: entry
    for entry in chair_raw["sentences"]
}

    eval_minigpt_metadata_only_from_dir(
        responses_dir=RESPONSES_DIR,
        output_dir=OUTPUT_DIR,
        chair_results=chair_results
    )