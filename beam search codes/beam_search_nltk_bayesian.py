import nltk
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import os
import sys
import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.mlp_deployment import FPAttentionClassifier

from collections import defaultdict, Counter

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import AutoTokenizer

import os
import matplotlib.pyplot as plt 
from llava.constants import IMAGE_TOKEN_INDEX
import numpy as np
import re
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


import re
from typing import List


def charpos_to_lemma(char_pos, word_spans):
    for word, lemma, start, end in word_spans:
        if char_pos == 0:
            return lemma
        else:
            if start-1 == char_pos:
                return lemma
    return None

def sort_select(scores):
    return np.sort(scores[0])[-200:]

def collect_scores_by_indices(score_records, llava_indices):
    collected = []
    for inds in llava_indices:
        for i in inds:
            if i < len(score_records):
                collected.append([i, sort_select(score_records[i].cpu().numpy())])
            break
    return collected

def topk_batch(attn, k):
    # attn: (H, N)
    k = min(k, attn.shape[-1])
    vals, idx = torch.topk(attn, k=k, dim=-1)
    return vals, idx


def sentence_evaluator(
    candidate_attentions,
    candidate_logits,
    classifier,
    token_offset,
    tokenizer,
    output_ids=None,
    image_start_idx=35,
    image_length=576,
    topk=20,
    layer_start=0,
    layer_end=32,
    use_nltk=False,  
):
    
    samples_4_classifier = []
    
    num_layers, num_heads, TOPK = 32, 32, 20
    
    sentence_len = len(candidate_attentions)
    
    scores = candidate_logits
    # tp_scores = collect_scores_by_indices(scores, tp_llava_indices)
    
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

        

    def word_to_first_token_id(word):
        ids = tokenizer(
            word,
            add_special_tokens=False
        )["input_ids"]
        return ids[0] if len(ids) > 0 else None


    token_ids_set = set()
    for w in words:
        tid = word_to_first_token_id(w)
        if tid is not None:
            token_ids_set.add(tid)


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

    fast_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer.name_or_path,
            use_fast=True
        )
    
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

            
            sample_dict = {
                "token_idx": idx,
                "topk_attn": topk_image_values.copy(),
                "topk_attn_next": topk_image_values_next.copy(),
                "logits": None,
                "total_reps": total_reps,
                "rep_num": occurrence_num,
                "token_id": token_id
                }


            samples_4_classifier.append(sample_dict)
        

    preds = classifier.predict([samples_4_classifier])

    return preds




def _clone_past_key_values(past_key_values):
    """Clone a tuple of past_key_values tensors safely (detach + clone)."""
    if past_key_values is None:
        return None
    return tuple(tuple(p.detach().clone() for p in layer) for layer in past_key_values)

def _detach_and_clone_model_kwargs(model_kwargs, device=None):
    """Shallow copy model_kwargs but deep-clone its tensor values."""
    mk = {}
    for k, v in model_kwargs.items():
        if k == "past_key_values":
            mk[k] = _clone_past_key_values(v)
        elif isinstance(v, torch.Tensor):
            mk[k] = v.detach().clone()
        elif isinstance(v, (list, tuple)):
            mk[k] = type(v)(
                (item.detach().clone() if isinstance(item, torch.Tensor) else item)
                for item in v
            )
        else:
            mk[k] = copy.deepcopy(v)
    if device is not None and "past_key_values" in mk and mk["past_key_values"] is not None:
        mk["past_key_values"] = tuple(
            tuple(t.to(device) for t in layer) for layer in mk["past_key_values"]
        )
    return mk

def _assert_cache_alignment(model_kwargs, input_ids):
    """Check that cached sequence length matches current token count."""
    pk = model_kwargs.get("past_key_values", None)
    if pk is None:
        return
    cached_len = pk[0][0].shape[-2]
    if cached_len != input_ids.shape[1]:
        print(f"[WARN] Cache length {cached_len} != input tokens {input_ids.shape[1]}")



def sample_with_ensemble(
    self,
    input_ids: torch.LongTensor,
    ensemble_size: int = 10,
    pseudo_sentence_length: int = 20,
    search_start: int = 2,
    use_nltk: bool = True,        # for POS tagging
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = True,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
):
    """
    Ensemble-guided sampling with diversity encouragement and optional NLTK POS tagging.
    """
    classifier = getattr(self, "classifier", None)
    tokenizer = getattr(self, "tokenizer", None)
    # ---------- Setup ----------
    logits_processor = logits_processor or LogitsProcessorList()
    logits_warper = logits_warper or LogitsProcessorList()
    stopping_criteria = stopping_criteria or StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

    pad_token_id = pad_token_id or self.generation_config.pad_token_id
    eos_token_id = eos_token_id or self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device)

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # ---------- Initialize ----------
    final_input_ids = input_ids.clone()
    model_kwargs_main = model_kwargs.copy()
    finished = False
    total_generated = 0

    # ---------- Phase 1: Warm-up ----------
    while total_generated < search_start and not finished:
        model_inputs = self.prepare_inputs_for_generation(final_input_ids, **model_kwargs_main)
        outputs = self(**model_inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_scores = logits_processor(final_input_ids, next_token_logits)
        next_token_scores = logits_warper(final_input_ids, next_token_scores)
        probs = F.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        final_input_ids = torch.cat([final_input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs_main = self._update_model_kwargs_for_generation(
            outputs, model_kwargs_main, is_encoder_decoder=self.config.is_encoder_decoder
        )

        total_generated += 1
        if next_tokens[0].item() in eos_token_id:
            finished = True
        if streamer is not None:
            streamer.put(next_tokens.cpu())

    while not finished:
        candidate_outputs, candidate_attentions, candidate_logits, candidate_scores, candidate_model_kwargs = [], [], [], [], []

        for candidate_idx in range(ensemble_size):
            model_kwargs_tmp = _detach_and_clone_model_kwargs(model_kwargs_main, device=next(self.parameters()).device)
            input_tmp = final_input_ids.clone()
            decoder_attn_tmp, decoder_lgt_tmp, token_count, eos_hit = [], [], 0, False

            while token_count < pseudo_sentence_length and not eos_hit:
                model_inputs = self.prepare_inputs_for_generation(input_tmp, **model_kwargs_tmp)
                outputs = self(**model_inputs, return_dict=True, output_attentions=True)

                next_token_logits = outputs.logits[:, -1, :]
                next_token_scores = logits_processor(input_tmp, next_token_logits)
                next_token_scores = logits_warper(input_tmp, next_token_scores)
                probs = F.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # decoder_lgt_tmp.append(outputs.scores)
                decoder_lgt_tmp.append(next_token_scores)

                input_tmp = torch.cat([input_tmp, next_tokens[:, None]], dim=-1)
                decoder_attn_tmp.append(
                    outputs.cross_attentions if self.config.is_encoder_decoder else outputs.attentions
                )
                model_kwargs_tmp = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs_tmp, is_encoder_decoder=self.config.is_encoder_decoder
                )

                token_count += 1
                if next_tokens[0].item() in eos_token_id:
                    eos_hit = True


            candidate_outputs.append(input_tmp)
            candidate_attentions.append(decoder_attn_tmp)
            candidate_logits.append(decoder_lgt_tmp)
            candidate_model_kwargs.append(model_kwargs_tmp)

        if len(candidate_outputs) == 0:
            candidate_outputs.append(input_tmp)
            candidate_attentions.append(decoder_attn_tmp)
            candidate_model_kwargs.append(model_kwargs_tmp)

        # ---------- Evaluate candidates ----------
        for i in range(len(candidate_outputs)):
            candidate_scores.append(
                sentence_evaluator(candidate_attentions[i], candidate_logits[i], classifier, total_generated, tokenizer, candidate_outputs[i], use_nltk=use_nltk)
            )

        total_generated += pseudo_sentence_length
        best_idx = int(torch.tensor(candidate_scores).argmax())

        final_input_ids = candidate_outputs[best_idx]
        model_kwargs_main = _detach_and_clone_model_kwargs(
            candidate_model_kwargs[best_idx], device=next(self.parameters()).device
        )

        _assert_cache_alignment(model_kwargs_main, final_input_ids)

        best_tokens = candidate_outputs[best_idx][:, -pseudo_sentence_length:]
        if any(t.item() in eos_token_id for t in best_tokens[0]) or stopping_criteria(final_input_ids, None):
            finished = True

        if streamer is not None:
            streamer.put(best_tokens.cpu())

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        return SampleDecoderOnlyOutput(sequences=final_input_ids, attentions=None, hidden_states=None)
    else:
        return final_input_ids


def evolve_beam_search():
    transformers.generation.utils.GenerationMixin.sample = sample_with_ensemble
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = sample_with_ensemble


