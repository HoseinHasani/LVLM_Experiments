import copy
import warnings
from typing import List, Optional, Union
import os
import sys
import torch

from data_extraction_utils import sentence_data_extraction

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

import torch.nn.functional as F
import os
import numpy as np




def halluscope_score(preds, alpha=0.5, beta=0.3):
    """
    preds: list of dicts returned by halluscope.predict()

    Scoring rule:
        score = alpha * real_num - hall_num - beta * hall_tot_repeat

    where (ONLY first occurrences are counted):
        real_num = count(final_pred == 1)
        hall_num = count(final_pred == 0)
        hall_tot_repeat = sum(len(all_occurrence_indices)) for hallucinated nouns
    """

    real_num = 0
    hall_num = 0
    hall_tot_repeat = 0

    for p in preds:
        if not p.get("is_first_occ", False):
            continue
        
        if "is_ms_object" in p.keys():
            if not p["is_ms_object"]:
                print(".")
                continue

        pred = p.get("final_pred", None)
        if pred is None:
            continue

        if pred == 1:
            real_num += 1
        elif pred == 0:
            hall_num += 1
            hall_tot_repeat += len(p.get("all_occurrence_indices", []))

    score = alpha * real_num - hall_num - beta * hall_tot_repeat
    
    return score



def sentence_evaluator(
    candidate_attentions,
    candidate_logits,
    halluscope,
    token_offset,
    tokenizer,
    fast_tokenizer,
    output_ids,
    image_start_idx=35,
    image_length=576,
    topk=20,
    ms_coco_obj=None,
    skip_end=True
):
    
    
    samples_4_classification = sentence_data_extraction(candidate_attentions,
                                                        candidate_logits,
                                                        tokenizer,
                                                        fast_tokenizer,
                                                        output_ids,
                                                        token_offset,
                                                        mscoco_objects=ms_coco_obj,
                                                        skip_end=skip_end
                                                        )

    preds = halluscope.predict(samples_4_classification)
    score = halluscope_score(preds)
    return score




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
    pseudo_sentence_length: int = 10,
    search_start: int = 2,
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
    mscoco_objects= np.load("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/rz15dire/Ensemble/experiments/eval_new/mscoco_objects.npy")

    # mscoco_objects = None
    classifier = getattr(self, "classifier", None)
    tokenizer = getattr(self, "tokenizer", None)
    fast_tokenizer = getattr(self, "fast_tokenizer", None)
    
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
    ensemble_iter = 0
    prompt_len = input_ids.shape[1]

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
        
        ensemble_iter += 1

        if ensemble_iter == 1:
            current_pseudo_length = 3 * pseudo_sentence_length
        elif ensemble_iter == 2:
            current_pseudo_length = 1 * pseudo_sentence_length
        else:
            current_pseudo_length = pseudo_sentence_length


        candidate_outputs, candidate_attentions, candidate_logits, candidate_scores, candidate_model_kwargs = [], [], [], [], []

        for candidate_idx in range(ensemble_size):
            model_kwargs_tmp = _detach_and_clone_model_kwargs(model_kwargs_main, device=next(self.parameters()).device)
            input_tmp = final_input_ids.clone()
            decoder_attn_tmp, decoder_lgt_tmp, token_count, eos_hit = [], [], 0, False

            while token_count < current_pseudo_length+1 and not eos_hit:
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
                if token_count < current_pseudo_length:
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
                sentence_evaluator(candidate_attentions[i], candidate_logits[i], classifier, total_generated,
                                   tokenizer, fast_tokenizer, candidate_outputs[i][:,prompt_len:],ms_coco_obj=mscoco_objects, 
                                   skip_end=len(candidate_attentions)==(current_pseudo_length+1))
            )


        best_idx = int(torch.tensor(candidate_scores).argmax())
        total_generated += min(current_pseudo_length, len(candidate_attentions[best_idx]))

        
        final_input_ids = candidate_outputs[best_idx]
        eos_cond = any(t.item() in eos_token_id for t in final_input_ids[0])

        if not eos_cond:
            final_input_ids = final_input_ids[:,:-1]

        model_kwargs_main = _detach_and_clone_model_kwargs(
            candidate_model_kwargs[best_idx], device=next(self.parameters()).device
        )

        _assert_cache_alignment(model_kwargs_main, final_input_ids)
        
        best_tokens = candidate_outputs[best_idx][:, -current_pseudo_length:]

        if not eos_cond:
            best_tokens = best_tokens[:,:-1]

        if eos_cond or stopping_criteria(final_input_ids, None):
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


