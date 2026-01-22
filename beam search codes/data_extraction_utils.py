import nltk
import spacy
from collections import defaultdict, Counter
import numpy as np
import torch
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")

def charpos_to_lemma(char_pos, word_spans):
    for word, lemma, start, end in word_spans:
        if char_pos == 0:
            return lemma
        elif start - 1 == char_pos:
            return lemma
    return None

def sort_select(logits):
    return np.sort(logits.cpu())[-200:]


def topk_batch(attn, k):
    k = min(k, attn.shape[-1])
    vals, idx = torch.topk(attn, k=k, dim=-1)
    return vals, idx


PHYSICAL_OBJECT_ROOTS = {
    "artifact.n.01",
    "instrumentality.n.03",
    "physical_entity.n.01",
    "object.n.01",
    "whole.n.02",  # optional for "image" or composite objects
}

def is_object_noun(token):
    """
    Return True if token is a noun and corresponds to a physical object in WordNet.
    Traverses hypernyms recursively.
    """
    if token.pos_ not in {"NOUN", "PROPN"}:
        return False

    # lemma = token.lemma_.lower()
    # synsets = wn.synsets(lemma, pos=wn.NOUN)
    # if not synsets:
    #     return True  # fallback: unknown nouns, assume object

    # for syn in synsets:
    #     if _has_physical_hypernym(syn):
    #         return True
    return True

def _has_physical_hypernym(syn):
    """Recursively check if synset has a hypernym in PHYSICAL_OBJECT_ROOTS"""
    if syn.name() in PHYSICAL_OBJECT_ROOTS:
        return True
    for hyper in syn.hypernyms():
        if _has_physical_hypernym(hyper):
            return True
    return False

def sentence_data_extraction(
    candidate_attentions,
    candidate_logits,
    tokenizer,
    fast_tokenizer,
    output_ids,
    token_offset=0,
    image_id=0,
    image_start_idx=35,
    image_length=576,
    topk=20,
    mscoco_objects=None,
):
    """
    Extract per-token data for classification, only for object-related tokens.
    """
    samples_4_classification = []

    num_layers, num_heads, TOPK = 32, 32, topk
    sentence_len = len(candidate_attentions)

    # Get generated text
    current_ids = output_ids
    gen_text = tokenizer.batch_decode(current_ids, skip_special_tokens=True)[0]

    # Tokenize and create word spans
    doc = nlp(gen_text)
    word_spans = [(token.text, token.lemma_.lower(), token.idx, token.idx + len(token.text))
                  for token in doc if not token.is_space and not token.is_punct]

    lemmas = [lemma for _, lemma, _, _ in word_spans]
    lemma_total_counter = Counter(lemmas)
    lemma_occurrence_counter = defaultdict(int)
    lemma_to_indices = defaultdict(list)

    # Encode with fast tokenizer to get offsets
    enc = fast_tokenizer(gen_text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # Collect lemma → token indices
    for idx, (start_char, end_char) in enumerate(offsets):
        if start_char == end_char:
            continue
        lemma = charpos_to_lemma(start_char, word_spans)
        if lemma is None:
            continue
        lemma_to_indices[lemma].append(idx)

    # Compute first subtoken map for all words
    words_lower = [token.text.lower() for token in doc if not token.is_space and not token.is_punct]

    seen_lemmas_per_word = defaultdict(bool)
    skip_next = set()

    # --- Identify object nouns ---
    object_nouns = [t for t in doc if is_object_noun(t)]
    object_char_spans = [(t.idx, t.idx + len(t.text)) for t in object_nouns]

    # Build subtoken mask
    object_token_mask = []
    object_word_starts = {start-1 for start, end in object_char_spans}
    for start, end in offsets:
        if start == end:
            object_token_mask.append(False)
            continue
        object_token_mask.append(start in object_word_starts)

    # --- Loop through tokens ---
    for idx, (token_id, (start_char, end_char)) in enumerate(zip(token_ids, offsets)):
        if not object_token_mask[idx]:
            continue  # skip non-object tokens

        lemma = charpos_to_lemma(start_char, word_spans)
        if lemma is None:
            continue

        if idx in skip_next:
            continue

        # Update occurrence counts
        if not seen_lemmas_per_word.get(start_char, False):
            lemma_occurrence_counter[lemma] += 1
            seen_lemmas_per_word[start_char] = True

        occurrence_num = lemma_occurrence_counter[lemma]
        total_reps = lemma_total_counter[lemma]

        # Extract top-k image attentions and sums
        topk_image_indices = np.zeros((num_layers, num_heads, TOPK), dtype=np.int64)
        topk_image_values  = np.zeros((num_layers, num_heads, TOPK), dtype=np.float32)
        topk_image_indices_next = np.zeros((num_layers, num_heads, TOPK), dtype=np.int64)
        topk_image_values_next  = np.zeros((num_layers, num_heads, TOPK), dtype=np.float32)
        image_sum = np.zeros((num_layers, num_heads), dtype=np.float32)

        for layer_idx in range(num_layers):
            attn_vec = candidate_attentions[idx][layer_idx][...,-1,:]  # (H, Q, K)
            img_attn = attn_vec[..., image_start_idx:image_start_idx + image_length][0]

            image_sum[layer_idx] = img_attn.sum(dim=-1).cpu().numpy()
            vals, idxs = topk_batch(img_attn, TOPK)
            topk_image_indices[layer_idx] = idxs.cpu().numpy()
            topk_image_values[layer_idx]  = vals.cpu().numpy()

            # next token attentions
            if idx + 1 < len(candidate_attentions):
                attn_vec_nxt = candidate_attentions[idx+1][layer_idx][...,-1,:]
                img_attn_nxt = attn_vec_nxt[..., image_start_idx:image_start_idx + image_length][0]
                vals_nxt, idxs_nxt = topk_batch(img_attn_nxt, TOPK)
                topk_image_indices_next[layer_idx] = idxs_nxt.cpu().numpy()
                topk_image_values_next[layer_idx]  = vals_nxt.cpu().numpy()

        # Select logits
        logit = sort_select(candidate_logits[idx])

        all_occurrence_indices = lemma_to_indices[lemma].copy() if occurrence_num == 1 and total_reps > 1 else []

        # Build sample dict
        sample_dict = {
            "sentence_idx": image_id,
            "caption": gen_text,
            "token_idx": idx,
            "topk_attn": topk_image_values.copy(),
            "topk_attn_next": topk_image_values_next.copy(),
            "logits": logit,
            "total_reps": total_reps,
            "rep_num": occurrence_num,
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "is_first_occ": occurrence_num == 1,
            "lemma": lemma,
            "all_occurrence_indices": all_occurrence_indices,
        }
        
        if mscoco_objects:
            sample_dict["is_ms_object"] = lemma in mscoco_objects

        samples_4_classification.append(sample_dict)

    return samples_4_classification
