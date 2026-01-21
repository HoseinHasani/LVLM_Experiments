import nltk
import copy
import os
import sys
import torch
from data_extraction_utils import sentence_data_extraction
from collections import defaultdict, Counter
import numpy as np
import re
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])



def charpos_to_lemma(char_pos, word_spans):
    for word, lemma, start, end in word_spans:
        if char_pos == 0:
            return lemma
        else:
            if start-1 == char_pos:
                return lemma
    return None

def sort_select(logits):
    return np.sort(logits)[-200:]



def first_subtoken_map(words, tokenizer):
    """
    Returns a dict mapping each word to a set of plausible first subtoken IDs,
    including common inflectional variants (plural, -ed, -ing).
    """
    mapping = {}
    for word in words:
        token_ids_set = set()
        # include original word and common variants
        variants = [word, word + 's', word + 'es', word + 'ed', word + 'ing']
        for var in variants:
            ids = tokenizer(var, add_special_tokens=False)["input_ids"]
            if len(ids) > 0:
                token_ids_set.add(ids[0])
        mapping[word] = token_ids_set
    return mapping
    


def topk_batch(attn, k):
    # attn: (H, N)
    k = min(k, attn.shape[-1])
    vals, idx = torch.topk(attn, k=k, dim=-1)
    return vals, idx


def sentence_data_extraction(
    candidate_attentions,
    candidate_logits,
    halluscope,
    token_offset,
    tokenizer,
    fast_tokenizer,
    output_ids,
    image_start_idx=35,
    image_length=576,
    topk=20):
    
    samples_4_classification = []
    
    num_layers, num_heads, TOPK = 32, 32, topk
    
    sentence_len = len(candidate_attentions)
    
    
    token_len = output_ids.shape[1]
    output_attentions = candidate_attentions

    # Decode generated text
    # gen_text = tokenizer.batch_decode(
    #     output_ids[:, -sentence_len:], skip_special_tokens=True
    # )[0]

    current_ids = output_ids[:, -(sentence_len+token_offset):]
    gen_text = tokenizer.batch_decode(
        current_ids, skip_special_tokens=True
    )[0]
    
    subwords = [tokenizer.decode([tid]) for tid in current_ids[0]]

    words = nltk.word_tokenize(gen_text.lower())
    tagged_sent = nltk.pos_tag(words)
    nouns = [[word,word] for word, tag in tagged_sent if "NN" in tag]

    doc = nlp(gen_text)


    lemmas = []
    word_spans = []  # (lemma, start_char, end_char)

    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemmas.append(token.lemma_.lower())
        word_spans.append((token.text, token.lemma_.lower(), token.idx, token.idx + len(token.text)))

    lemma_total_counter = Counter(lemmas)
    
    lemma_occurrence_counter = defaultdict(int)
    lemma_to_indices = defaultdict(list)



    selected_noun_indices = []
    

    # --- tokenizer offsets for generated text ---

    
    enc = fast_tokenizer(
        gen_text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]  # list of (start_char, end_char) per subtoken

    # --- FIRST PASS: collect lemma → token indices ---
    for idx, (start_char, end_char) in enumerate(offsets):
        if start_char == end_char:
            continue

        lemma = charpos_to_lemma(start_char, word_spans)
        if lemma is None:
            continue

        lemma_to_indices[lemma].append(idx)



    gen_text_first_subtoken_map = first_subtoken_map(words, tokenizer)


    gen_text_lemmas = set()
    for w in words:
        doc_w = nlp(w)
        for t in doc_w:
            if not t.is_space and not t.is_punct:
                gen_text_lemmas.add(t.lemma_.lower())


    selected_noun_positions = set()

    seen_lemmas_per_word = defaultdict(bool)
    skip_next = set()  # indices to skip
    

    for idx, (token_id, (start_char, end_char)) in enumerate(zip(token_ids, offsets)):
            if tokenizer.decode([token_id]) == "cars":
                print("hi")
            if start_char == end_char:
                continue  # whitespace or special

            lemma = charpos_to_lemma(start_char, word_spans)
            if lemma is None:
                continue

            # Skip subtokens that are in skip_next
            if idx in skip_next:
                continue

            # Only increment occurrence on the first subtoken of the word
            if not seen_lemmas_per_word.get(start_char, False):
                lemma_occurrence_counter[lemma] += 1
                seen_lemmas_per_word[start_char] = True


            occurrence_num = lemma_occurrence_counter[lemma]
            total_reps = lemma_total_counter[lemma]

                # check hallucinated words
            for word, first_id in gen_text_first_subtoken_map.items():
                word_lemma = nlp(word)[0].lemma_.lower()  # first lemma of word
                if lemma == word_lemma and token_id in first_id:
                    selected_noun_positions.add(idx)


            if idx in selected_noun_positions:
                token_type = "noun"
                selected_noun_indices.append([idx])
                word_ids = tokenizer(lemma, add_special_tokens=False)["input_ids"]
                skip_next.update(idx + i for i in range(1, len(word_ids)+1))

            
            logit = sort_select(candidate_logits[idx])

            topk_image_indices = np.zeros((num_layers, num_heads, TOPK), dtype=np.int64)
            topk_image_values  = np.zeros((num_layers, num_heads, TOPK), dtype=np.float32)

            topk_text_indices  = np.zeros((num_layers, num_heads, TOPK), dtype=np.int64)
            topk_text_values   = np.zeros((num_layers, num_heads, TOPK), dtype=np.float32)

            image_sum = np.zeros((num_layers, num_heads), dtype=np.float32)
            text_sum  = np.zeros((num_layers, num_heads), dtype=np.float32)

            topk_image_indices_next = np.zeros((num_layers, num_heads, TOPK), dtype=np.int64)
            topk_image_values_next  = np.zeros((num_layers, num_heads, TOPK), dtype=np.float32)

            topk_text_indices_next  = np.zeros((num_layers, num_heads, TOPK), dtype=np.int64)
            topk_text_values_next   = np.zeros((num_layers, num_heads, TOPK), dtype=np.float32)

            for layer_idx in range(num_layers):
                target_attn_vec = output_attentions[idx][layer_idx][...,-1, :]  #latest query           # (H, Q, K)


                img_attn = target_attn_vec[..., image_start_idx:image_start_idx + image_length][0]

                text_attn_before_img = target_attn_vec[...,:image_start_idx]
                text_attn_after_img = target_attn_vec[...,image_start_idx + image_length:]
                txt_attn =torch.cat([text_attn_before_img, text_attn_after_img], dim=-1)[0]                 # (H, T)

                # sums
                image_sum[layer_idx] = img_attn.sum(dim=-1).cpu().numpy()
                text_sum[layer_idx]  = txt_attn.sum(dim=-1).cpu().numpy()

                # top-k
                img_vals, img_idx = topk_batch(img_attn, TOPK)
                txt_vals, txt_idx = topk_batch(txt_attn, TOPK)
                #txt index inja nesbi nis 
                topk_image_indices[layer_idx] = img_idx.cpu().numpy()
                topk_image_values[layer_idx]  = img_vals.cpu().numpy()

                topk_text_indices[layer_idx]  = txt_idx.cpu().numpy()
                topk_text_values[layer_idx]   = txt_vals.cpu().numpy()

                # ---------- NEXT TOKEN ----------
                if idx+1 < len(output_attentions):
                    target_attn_vec_nxt = output_attentions[idx+1][layer_idx][...,-1, :]  #latest query           # (H, Q, K)

                    img_attn_nxt = target_attn_vec_nxt[..., image_start_idx:image_start_idx + image_length][0]

                    text_attn_before_img_nxt = target_attn_vec_nxt[...,:image_start_idx]
                    text_attn_after_img_nxt = target_attn_vec_nxt[...,image_start_idx + image_length:]
                    txt_attn_nxt =torch.cat([text_attn_before_img_nxt, text_attn_after_img_nxt], dim=-1)[0]                 # (H, T)

                    # top-k
                    img_vals_nxt, img_idx_nxt = topk_batch(img_attn_nxt, TOPK)
                    txt_vals_nxt, txt_idx_nxt = topk_batch(txt_attn_nxt, TOPK)
                    #txt index inja nesbi nis 
                    topk_image_indices_next[layer_idx] = img_idx_nxt.cpu().numpy()
                    topk_image_values_next[layer_idx]  = img_vals_nxt.cpu().numpy()

                    topk_text_indices_next[layer_idx]  = txt_idx_nxt.cpu().numpy()
                    topk_text_values_next[layer_idx]   = txt_vals_nxt.cpu().numpy()

            
            if occurrence_num == 1 and total_reps > 1:
                all_occurrence_indices = lemma_to_indices[lemma].copy()
            else:
                all_occurrence_indices = []
                
            sample_dict = {
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


            samples_4_classification.append(sample_dict)
            
    return samples_4_classification
