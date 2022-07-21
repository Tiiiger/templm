import torch
import torch.nn as nn
import torch.distributed as dist

import warnings
import logging
from tqdm.auto import trange, tqdm
import re

from transformers import BartForConditionalGeneration, BeamSearchScorer
from transformers.generation_beam_search import BeamHypotheses
from dataclasses import dataclass, field
from typing import Optional
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.file_utils import is_offline_mode, ModelOutput
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


from transformers.generation_utils import (
    validate_stopping_criteria,
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
)
from collections import defaultdict, UserDict
import utils

logger = logging.getLogger(__name__)


class TemplateSearchBART(BartForConditionalGeneration):
    """
    This class is created to modify the `generate` function of BART.
    Unfortunately most code is duplicated from the parent class. Changes are marked.

    This generate function now:
    1. returns the most `num_beams` probable templates computed on all the input_ids
    2. computes everything in batches under the hood

    """

    def init_for_template_search(
        self, tokenizer, full_data_field_dicts, hallucinate_aug
    ):
        self.tokenizer = tokenizer
        self.hallucinate_aug = hallucinate_aug

    @torch.no_grad()
    def generate(
        self,
        # CHANGE: add in `field_dicts` and `tokenizer` as required arguments
        input_ids,
        field_dicts,
        # CHANGE: add inference batch size for internal batching,
        inference_batch_size,
        # CHANGE: add option to recompute log probs which save memory
        recompute_log_prob,
        # CHANGE: add in weighting,
        input_weighting,
        mode="template_search",
        search_crit="prob",
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids=None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        verbose=False,
        **model_kwargs,
    ):
        # set init values
        if max_length is None and max_new_tokens is None:
            # Both are None, default
            max_length = self.config.max_length
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set but they serve the same purpose.",
                UserWarning,
            )

        max_length = max_length if max_length is not None else self.config.max_length
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = (
            num_beam_groups
            if num_beam_groups is not None
            else self.config.num_beam_groups
        )
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences
            if num_return_sequences is not None
            else self.config.num_return_sequences
        )

        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        bos_token_id = (
            bos_token_id if bos_token_id is not None else self.config.bos_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )

        output_scores = (
            output_scores if output_scores is not None else self.config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.config.return_dict_in_generate
        )
        # CHANGE: force `inference_batch_size` to be multiples of `num_beams`
        inference_batch_size = (inference_batch_size // num_beams) * num_beams

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states

        if input_ids is None and "inputs_embeds" not in model_kwargs:
            # init `input_ids` with bos_token_id
            input_ids = self._prepare_input_ids_for_generation(
                bos_token_id, model_kwargs.get("encoder_outputs")
            )

        if model_kwargs.get("attention_mask", None) is None:
            # init `attention_mask` depending on `pad_token_id`
            model_kwargs[
                "attention_mask"
            ] = self._prepare_attention_mask_for_generation(
                input_ids, pad_token_id, eos_token_id
            )

        # special case if pad_token_id is not defined
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation."
            )
            pad_token_id = eos_token_id

        # Storing encoder_input_ids for logits_processor that could use them
        encoder_input_ids = input_ids if self.config.is_encoder_decoder else None

        if self.config.is_encoder_decoder:
            # add encoder_outputs to model_kwargs
            # CHANGE: computing the encoder states in batch
            all_last_hidden_states = []
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (
                    argument.startswith("decoder_") or argument.startswith("cross_attn")
                )
            }
            for batch_start in range(0, input_ids.size(0), inference_batch_size):
                batch_attention_mask = encoder_kwargs["attention_mask"][
                    batch_start : batch_start + inference_batch_size
                ].cuda()
                batch_input_ids = input_ids[
                    batch_start : batch_start + inference_batch_size
                ].cuda()
                batch_encoder_outputs = encoder(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    return_dict=True,
                )
                all_last_hidden_states.append(
                    batch_encoder_outputs.last_hidden_state.cpu()
                )
            all_last_hidden_states = torch.cat(all_last_hidden_states, dim=0)
            batch_encoder_outputs.last_hidden_state = all_last_hidden_states
            model_kwargs["encoder_outputs"] = batch_encoder_outputs

            # set input_ids as decoder_input_ids
            if "decoder_input_ids" in model_kwargs:
                input_ids = model_kwargs.pop("decoder_input_ids")
            else:
                input_ids = self._prepare_decoder_input_ids_for_generation(
                    input_ids,
                    decoder_start_token_id=decoder_start_token_id,
                    bos_token_id=bos_token_id,
                )

            if "encoder_outputs" not in model_kwargs or not isinstance(
                model_kwargs["encoder_outputs"], ModelOutput
            ):
                raise ValueError(
                    "Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`."
                )

        if input_ids.shape[-1] >= max_length:
            input_ids_string = (
                "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            )
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # determine generation mode
        is_greedy_gen_mode = (
            (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        )
        is_sample_gen_mode = (
            (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        )
        is_beam_gen_mode = (
            (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        )
        is_beam_sample_gen_mode = (
            (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        )
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError(
                "`num_beam_groups` has to be smaller or equal to `num_beams`"
            )
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=encoder_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
        )

        cur_len = input_ids.shape[-1]
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length,
            max_time=max_time,
            max_new_tokens=max_new_tokens,
            start_length=cur_len,
        )
        # CHANGE: only support beam search so other modes are invalid
        # if is_beam_gen_mode:
        batch_size = input_ids.shape[0]

        length_penalty = (
            length_penalty if length_penalty is not None else self.config.length_penalty
        )
        early_stopping = (
            early_stopping if early_stopping is not None else self.config.early_stopping
        )

        if num_return_sequences > num_beams:
            raise ValueError(
                "`num_return_sequences` has to be smaller or equal to `num_beams`."
            )

        if stopping_criteria.max_length is None:
            raise ValueError("`max_length` needs to be a stopping_criteria for now.")

        # CHANGE: we only keep one beam for the template on cpu
        if mode == "beam_search":
            beam_scorer = BeamSearchScorer(
                batch_size=input_ids.size(0),
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
        elif mode == "template_search":
            beam_scorer = TemplateSearchScorer(
                batch_size=1,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
        else:
            raise ValueError
        # interleave with `num_beams`
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids,
            expand_size=num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        if mode == "template_search":
            return self.template_search(
                input_ids,
                field_dicts,
                inference_batch_size,
                recompute_log_prob,
                input_weighting,
                search_crit,
                beam_scorer,
                max_length=max_length,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                verbose=verbose,
                **model_kwargs,
            )
        elif mode == "beam_search":
            return self.beam_search(
                input_ids,
                beam_scorer,
                max_length=max_length,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def assign_nonterminal_logits(
        self,
        next_token_scores,
        beam_expanded_last_latent_tokens,
        beam_expanded_last_surface_tokens,
        beam_expanded_field_dicts,
        beam_expanded_field_selection_dicts,
    ):
        # CHANGE: this loops distributes the logits from terminal tokens to non-terminal tokens
        for i in range(next_token_scores.size(0)):
            last_latent_token = beam_expanded_last_latent_tokens[i].item()
            last_surface_token = beam_expanded_last_surface_tokens[i].item()
            # CONDITION: this is not None when last_latent_token is a field_token and is not the end of a field
            next_special_token = self.tokenizer.next_field_token_map.get(
                last_latent_token, None
            )
            # manually cutoff field tokens that are too long
            if last_latent_token in self.tokenizer.field_cutoff_set:
                next_special_token = None
            sen_field_dict = beam_expanded_field_dicts[i]
            sen_field_selection_dict = beam_expanded_field_selection_dicts[i]
            assert not self.hallucinate_aug, "Hallucination Aug nor supported yet."

            if next_special_token is not None:
                # CONDITION: the previous token is not the last non-terminal token of the field
                if next_special_token in sen_field_dict:
                    # We identify the corresponding terminal token and get its logits
                    if last_latent_token not in sen_field_selection_dict:
                        raise ValueError(
                            "Field not selected but appear in previous tokens."
                        )
                    previous_selected_indices = sen_field_selection_dict[
                        last_latent_token
                    ]
                    temp_selected_indices = [
                        j
                        for j in previous_selected_indices
                        if sen_field_dict[next_special_token][j] != -1
                    ]
                    candidate_tokens = [
                        sen_field_dict[next_special_token][j]
                        for j in temp_selected_indices
                    ]
                    if len(candidate_tokens) == 0:
                        # CONDITION: the previous token is the last non-terminal of the field
                        next_token_scores[i] = -100
                        next_token_scores[i, next_special_token] = 0
                        sen_field_selection_dict[next_special_token] = (
                            previous_selected_indices[0],
                        )
                        continue
                    else:
                        previous_selected_indices = temp_selected_indices
                    candidate_scores = next_token_scores[i, candidate_tokens]
                    field_selected_score, field_selected_idx = candidate_scores.max(
                        dim=0
                    )
                    selected_token = candidate_tokens[field_selected_idx.item()]
                    selected_indices = []
                    for j in previous_selected_indices:
                        if sen_field_dict[next_special_token][j] == selected_token:
                            selected_indices.append(j)
                    next_token_scores[i, next_special_token] = (
                        field_selected_score + 1e-6
                    )  # tie breaking
                    sen_field_selection_dict[next_special_token] = tuple(
                        selected_indices
                    )

                # CONDITION: the previous token is the last non-terminal of the field
                else:
                    # We force complete the rest by assigning all probability to the next
                    # non-terminal token (which doesn't have a corresponding terminal token)
                    next_token_scores[i] = -100
                    next_token_scores[i, next_special_token] = 0
            # CONDITION: the previous token is a terminal token or is the end of a field (the last non-terminal token)
            else:
                # We compute the probability of entering each field
                for field_start in self.tokenizer.field_start_set:
                    # CONDITION: input data has this field
                    if field_start in sen_field_dict:
                        candidate_scores = next_token_scores[
                            i, sen_field_dict[field_start]
                        ]
                        field_selected_score, field_selected_idx = candidate_scores.max(
                            dim=0
                        )
                        next_token_scores[i, field_start] = (
                            field_selected_score + 1e-6
                        )  # tie breaking

                        selected_token = sen_field_dict[field_start][
                            field_selected_idx.item()
                        ]
                        selected_indices = []
                        for j in range(len(sen_field_dict[field_start])):
                            if sen_field_dict[field_start][j] == selected_token:
                                selected_indices.append(j)
                        sen_field_selection_dict[field_start] = tuple(selected_indices)
                    # CONDITION: input data doesn't have this field
                    else:
                        pass
        return next_token_scores

    def _compute_next_token_scores_batched(
        self, model_inputs, batch_start, batch_size, logits_processor
    ):
        batch_decoder_input_ids = model_inputs["decoder_input_ids"][
            batch_start : batch_start + batch_size
        ].cuda()
        batch_encoder_last_state = (
            model_inputs["encoder_outputs"]
            .last_hidden_state[batch_start : batch_start + batch_size]
            .cuda()
        )
        batch_encoder_outputs = BaseModelOutput(
            last_hidden_state=batch_encoder_last_state
        )
        batch_attention_mask = model_inputs["attention_mask"][
            batch_start : batch_start + batch_size
        ].cuda()
        batch_model_inputs = {
            "decoder_input_ids": batch_decoder_input_ids,
            "encoder_outputs": batch_encoder_outputs,
            "attention_mask": batch_attention_mask,
        }
        batch_outputs = self(**batch_model_inputs, return_dict=True,)
        batch_logprobs = batch_outputs.logits.log_softmax(dim=-1)
        last_token_index = (
            batch_decoder_input_ids.ne(self.tokenizer.pad_token_id).sum(dim=-1) - 1
        )
        next_token_scores = torch.stack(
            [batch_logprobs[i, j] for i, j in enumerate(list(last_token_index))], dim=0,
        )
        next_token_scores = logits_processor(batch_decoder_input_ids, next_token_scores)
        # next_token_scores[:, self.tokenizer.all_field_tokens_list] = -100

        for i, last_token_i in enumerate(list(last_token_index)):
            batch_logprobs[i, last_token_i] = 0
        surface_sequence_scores = (
            batch_logprobs[:, :-1]
            .gather(index=batch_decoder_input_ids[:, 1:].unsqueeze(-1), dim=-1)
            .squeeze(-1)
        )
        surface_sequence_scores = surface_sequence_scores * (
            batch_decoder_input_ids[:, 1:].ne(self.tokenizer.pad_token_id)
        )
        surface_sequence_scores = surface_sequence_scores.sum(dim=-1)

        last_surface_ids = torch.stack(
            [
                batch_decoder_input_ids[i, j]
                for i, j in enumerate(list(last_token_index))
            ],
            dim=0,
        )
        return next_token_scores, last_surface_ids, surface_sequence_scores

    def compute_next_token_scores_batched(
        self,
        model_inputs,
        latent_input_ids,
        beam_expanded_field_dicts,
        beam_expanded_field_selection_dicts,
        inference_batch_size,
        logits_processor,
        surface_sequence_scores,
        input_weighting,
        search_crit,
        is_first_step=False,
    ):
        # CHANGE: logits are computed in batches.
        num_examples = model_inputs["decoder_input_ids"].size(0)
        last_latent_tokens = latent_input_ids[:, -1]
        num_beams = last_latent_tokens.size(0)
        total_batch_size = num_examples // num_beams
        beam_expanded_last_latent_tokens = (
            last_latent_tokens.unsqueeze(0)
            .expand(total_batch_size, num_beams)
            .reshape(-1)
        )

        all_next_token_scores = []
        all_surface_sequence_scores = []
        all_surface_sequence_scores_raw = []
        latent_sequence_scores = torch.zeros(
            num_beams * len(self.tokenizer), device="cuda"
        )

        # Precondition: inference_batch_size is a multiple of num_beams
        for batch_start in range(0, num_examples, inference_batch_size):
            (
                next_token_scores,
                batch_last_surface_ids,
                batch_surface_sequence_scores,
            ) = self._compute_next_token_scores_batched(
                model_inputs, batch_start, inference_batch_size, logits_processor
            )
            batch_last_latent_tokens = beam_expanded_last_latent_tokens[
                batch_start : batch_start + inference_batch_size
            ]
            real_batch_size = (
                min(batch_start + inference_batch_size, num_examples) - batch_start
            )
            batch_field_dicts = [
                beam_expanded_field_dicts[i]
                for i in range(batch_start, batch_start + real_batch_size)
            ]
            batch_field_selection_dicts = [
                beam_expanded_field_selection_dicts[i]
                for i in range(batch_start, batch_start + real_batch_size)
            ]
            next_token_scores = self.assign_nonterminal_logits(
                next_token_scores,
                batch_last_latent_tokens,
                batch_last_surface_ids,
                batch_field_dicts,
                batch_field_selection_dicts,
            )
            if is_first_step:
                batch_surface_sequence_scores = surface_sequence_scores[
                    batch_start : batch_start + inference_batch_size
                ].cuda()
            batch_surface_sequence_scores_raw = batch_surface_sequence_scores
            batch_surface_sequence_scores = next_token_scores + batch_surface_sequence_scores.unsqueeze(
                1
            ).expand_as(
                next_token_scores
            )
            if search_crit == "prob_lennorm":
                num_non_pad = (
                    model_inputs["decoder_input_ids"][
                        batch_start : batch_start + inference_batch_size
                    ]
                    .ne(self.tokenizer.pad_token_id)
                    .sum(dim=-1, keepdim=True)
                ).cuda()
                batch_surface_sequence_scores = (
                    batch_surface_sequence_scores / num_non_pad
                )
            batch_surface_sequence_scores = batch_surface_sequence_scores.view(
                real_batch_size // num_beams, num_beams * next_token_scores.size(-1),
            )
            batch_input_weighting = (
                input_weighting[
                    batch_start
                    // num_beams : (batch_start + real_batch_size)
                    // num_beams
                ]
                .to(latent_sequence_scores.device)
                .unsqueeze(1)
            )
            if search_crit in ["prob", "prob_lennorm"]:
                latent_sequence_scores += (
                    batch_surface_sequence_scores.exp() * batch_input_weighting
                ).sum(dim=0)
            elif search_crit == "log_prob":
                latent_sequence_scores += (
                    batch_surface_sequence_scores * batch_input_weighting
                ).sum(dim=0)
            else:
                raise ValueError("invalid search criterion", search_crit)

            if search_crit == "prob_lennorm":
                batch_surface_sequence_scores = batch_surface_sequence_scores.view(
                    real_batch_size, next_token_scores.size(-1),
                )
                batch_surface_sequence_scores = (
                    batch_surface_sequence_scores * num_non_pad
                )
                batch_surface_sequence_scores = batch_surface_sequence_scores.view(
                    real_batch_size // num_beams,
                    num_beams * next_token_scores.size(-1),
                )

            all_surface_sequence_scores.append(batch_surface_sequence_scores.cpu())
            all_surface_sequence_scores_raw.append(
                batch_surface_sequence_scores_raw.cpu()
            )
            all_next_token_scores.append(next_token_scores.cpu())
        surface_sequence_scores = torch.cat(all_surface_sequence_scores, dim=0)
        surface_sequence_scores_raw = torch.cat(all_surface_sequence_scores_raw, dim=0)
        if search_crit == "prob":
            latent_sequence_scores = latent_sequence_scores.log().cpu()
        elif search_crit == "log_prob":
            latent_sequence_scores = latent_sequence_scores.cpu()
        else:
            raise ValueError("invalid search criterion", search_crit)
        next_token_scores = torch.cat(all_next_token_scores)
        return (
            next_token_scores,
            surface_sequence_scores,
            surface_sequence_scores_raw,
            latent_sequence_scores,
        )

    def concat_new_ids(
        self,
        latent_input_ids,
        input_ids,
        all_word_log_probs,
        beam_idx,
        beam_next_tokens,
        next_token_scores,
        beam_expanded_field_dicts,
        beam_expanded_field_selection_dicts,
    ):
        # CHANGE:
        # This blocks concats newly generated tokens to existing beams
        # for the latent sequence, this is a simple concatenation
        # for the surface sequence, it:
        # 1. remove pad tokens in previous input_ids
        # 2. replace field tokens with the data
        # 3. pad to the same length
        # Invariance: <pad> only appear from the right
        latent_input_ids = torch.cat(
            [latent_input_ids[beam_idx], beam_next_tokens.unsqueeze(-1)], dim=-1
        )

        beam_next_tokens = (
            beam_next_tokens.unsqueeze(0)
            .expand(
                input_ids.size(0) // latent_input_ids.size(0), beam_next_tokens.size(0)
            )
            .reshape(-1, 1)
        )

        beam_idx = beam_idx.unsqueeze(0) + torch.arange(
            input_ids.size(0) // latent_input_ids.size(0)
        ).unsqueeze(1) * latent_input_ids.size(0)
        beam_idx = beam_idx.reshape(-1)

        next_token_scores = next_token_scores.reshape(-1, 1)
        all_word_log_probs = torch.cat(
            [all_word_log_probs[beam_idx], next_token_scores], dim=-1
        )

        list_input_ids = []
        for sen_ids in input_ids[beam_idx]:
            list_input_ids.append(
                sen_ids[: sen_ids.ne(self.tokenizer.pad_token_id).sum()]
            )

        beam_expanded_field_selection_dicts = [
            beam_expanded_field_selection_dicts[i].copy() for i in beam_idx.tolist()
        ]

        for i, next_token in enumerate(beam_next_tokens):
            sen_field_dict = beam_expanded_field_dicts[i]
            sen_field_selection_dict = beam_expanded_field_selection_dicts[i]
            int_next_token = next_token.item()
            if int_next_token not in self.tokenizer.all_field_tokens:
                list_input_ids[i] = torch.cat([list_input_ids[i], next_token])
            elif int_next_token in sen_field_dict:
                selected_idx = sen_field_selection_dict[int_next_token][0]
                int_next_surface_token = sen_field_dict[int_next_token][selected_idx]
                if int_next_surface_token != -1:
                    next_token = torch.LongTensor([int_next_surface_token]).to(
                        input_ids.device
                    )
                    list_input_ids[i] = torch.cat([list_input_ids[i], next_token])
                    if int_next_token in self.tokenizer.field_cutoff_set:
                        next_run_over_token = self.tokenizer.next_field_token_map.get(
                            int_next_token, None
                        )
                        run_over_tokens = []
                        while (
                            next_run_over_token is not None
                            and next_run_over_token in sen_field_dict
                        ):
                            int_next_surface_token = sen_field_dict[
                                next_run_over_token
                            ][selected_idx]
                            if int_next_surface_token != -1:
                                run_over_tokens.append(int_next_surface_token)
                                next_run_over_token = self.tokenizer.next_field_token_map.get(
                                    next_run_over_token, None
                                )
                            else:
                                break
                        if len(run_over_tokens) != 0:
                            next_tokens = torch.LongTensor(run_over_tokens).to(
                                input_ids.device
                            )
                            list_input_ids[i] = torch.cat(
                                [list_input_ids[i], next_tokens]
                            )

            elif (
                int_next_token in self.tokenizer.all_field_tokens
                and int_next_token not in self.tokenizer.field_start_set
            ):
                pass
            elif (
                int_next_token in self.tokenizer.all_field_tokens
                and int_next_token in self.tokenizer.field_start_set
            ):
                list_input_ids[i] = torch.cat([list_input_ids[i], next_token])
            else:
                raise ValueError("Impossible")

        # TODO: this syntax is disgusting
        input_ids = self.tokenizer.pad([{"input_ids": i} for i in list_input_ids])[
            "input_ids"
        ].to(input_ids.device)
        return (
            latent_input_ids,
            input_ids,
            all_word_log_probs,
            beam_expanded_field_selection_dicts,
        )

    def template_search(
        self,
        input_ids,
        field_dicts,
        inference_batch_size,
        recompute_log_prob,
        input_weighting,
        search_crit,
        beam_scorer,
        logits_processor=None,
        stopping_criteria=None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        verbose=False,
        **model_kwargs,
    ):
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever",
                UserWarning,
            )
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        output_scores = (
            output_scores if output_scores is not None else self.config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None

        num_beams = beam_scorer.num_beams
        batch_size = input_ids.size(0) // num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        # CHANGE: we keep track of both the surface sequence scores and the latent sequence scores (after averaging)
        surface_sequence_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device
        )
        surface_sequence_scores[:, 1:] = -1e9
        surface_sequence_scores = surface_sequence_scores.view(
            (batch_size * num_beams,)
        )

        latent_sequence_scores = torch.zeros(
            (num_beams,), dtype=torch.float, device=input_ids.device
        )
        latent_sequence_scores[1:] = -1e9

        # Duplicating field dict for each sequence on the beam
        beam_expanded_field_dicts = []
        for field_dict in field_dicts:
            for _ in range(num_beams):
                # NOTE: make a copy here so we won't modify real field_dict
                # this is used for the hallucination augmentation
                beam_expanded_field_dicts.append(field_dict.copy())

        # Initializing field selection dictionaries to help keep track
        beam_expanded_field_selection_dicts = [
            defaultdict(dict) for _ in range(surface_sequence_scores.size(0))
        ]

        latent_input_ids = input_ids[:num_beams]

        this_peer_finished = False  # used by synced_gpus only
        assert max_length is not None

        all_word_log_probs = torch.zeros(batch_size * num_beams, 1)

        # stats for debugging
        cutoff_counts = []
        step_input_ids = []
        step_latent_input_ids = []
        step_scores = []
        step_partial_surface_seq_scores = []
        step_partial_latent_seq_scores = []
        if verbose:
            bar = trange(max_length - 1, leave=False)
        else:
            bar = range(max_length)
        for loop_i in bar:

            # CHANGE: turning off caching to make implementation simple for now
            model_kwargs["past"] = None
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs["past_key_values"] = None

            (
                next_token_scores,
                surface_sequence_scores,
                surface_sequence_scores_raw,
                latent_sequence_scores,
            ) = self.compute_next_token_scores_batched(
                model_inputs,
                latent_input_ids,
                beam_expanded_field_dicts,
                beam_expanded_field_selection_dicts,
                inference_batch_size,
                logits_processor,
                surface_sequence_scores,
                input_weighting,
                search_crit,
                is_first_step=(loop_i == 0),
            )

            eos_scores = next_token_scores[:, self.tokenizer.eos_token_id]
            partial_surface_sequence_scores = eos_scores + surface_sequence_scores_raw
            partial_surface_sequence_scores = partial_surface_sequence_scores.view(
                -1, num_beams
            )
            step_partial_surface_seq_scores.append(partial_surface_sequence_scores)
            if search_crit == "prob":
                step_partial_latent_seq_scores.append(
                    partial_surface_sequence_scores.view(-1, num_beams)
                    .exp()
                    .mean(dim=0)
                    .log()
                )
            elif search_crit == "log_prob":
                step_partial_latent_seq_scores.append(
                    partial_surface_sequence_scores.view(-1, num_beams).mean(dim=0)
                )
            else:
                raise ValueError("impossible.")

            vocab_size = len(self.tokenizer)
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )

            latent_sequence_scores, next_tokens = torch.topk(
                latent_sequence_scores, 2 * num_beams, dim=0, largest=True, sorted=True,
            )
            expanded_next_tokens = next_tokens.unsqueeze(0).expand(
                surface_sequence_scores.size(0), next_tokens.size(0)
            )
            surface_sequence_scores = surface_sequence_scores.gather(
                index=expanded_next_tokens, dim=-1,
            )
            next_token_scores = next_token_scores.gather(
                index=expanded_next_tokens, dim=-1
            )
            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            # this block is unmodified from HF.
            beam_outputs = beam_scorer.process(
                latent_input_ids,
                latent_sequence_scores.unsqueeze(0),
                next_tokens.unsqueeze(0),
                next_indices.unsqueeze(0),
                input_ids.view(-1, num_beams, input_ids.size(-1)).transpose(0, 1),
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            latent_sequence_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            beam_token_idx = []
            for i, token_i in enumerate(list(next_tokens)):
                if token_i != self.tokenizer.eos_token:
                    beam_token_idx.append(i)
            beam_token_idx = torch.LongTensor(
                beam_token_idx[:num_beams], device=surface_sequence_scores.device
            )

            expanded_token_idx = beam_token_idx.unsqueeze(0).expand(
                surface_sequence_scores.size(0), beam_token_idx.size(0)
            )
            surface_sequence_scores = surface_sequence_scores.gather(
                index=expanded_token_idx, dim=1,
            )
            next_token_scores = next_token_scores.gather(
                index=expanded_token_idx, dim=1,
            )
            surface_sequence_scores = surface_sequence_scores.view(-1)

            (
                latent_input_ids,
                input_ids,
                all_word_log_probs,
                beam_expanded_field_selection_dicts,
            ) = self.concat_new_ids(
                latent_input_ids,
                input_ids,
                all_word_log_probs,
                beam_idx,
                beam_next_tokens,
                next_token_scores,
                beam_expanded_field_dicts,
                beam_expanded_field_selection_dicts,
            )

            if verbose:
                for i in range(2):
                    print(
                        "BEAM {} {:.2f}".format(i, latent_sequence_scores[i].item()),
                        self.tokenizer.decode(latent_input_ids[i]),
                    )
                    print(
                        "BEAM {} {:.2f}".format(i, surface_sequence_scores[i].item()),
                        self.tokenizer.decode(input_ids[i]),
                    )
            # calculate the number of sequences that take up 90% of the voting
            beam_raw_probs = surface_sequence_scores_raw.view(-1, num_beams).exp()
            sorted_first_beam_raw_probs = torch.sort(
                beam_raw_probs[:, 0], 0, descending=True
            )[0].cumsum(dim=0)
            first_beam_raw_probs_ratio = (
                sorted_first_beam_raw_probs / sorted_first_beam_raw_probs[-1]
            )
            count = (first_beam_raw_probs_ratio < 0.9).long().sum().item()
            if verbose:
                print("# of inputs that takes up 90% of the voting: ", count)
            cutoff_counts.append(count)
            step_input_ids.append(input_ids)
            step_latent_input_ids.append(latent_input_ids)
            step_scores.append(beam_raw_probs)

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        # the return logic is minimally modified to return the most probably latent sequence
        sequence_outputs = beam_scorer.finalize(
            latent_input_ids,
            latent_sequence_scores,
            next_tokens,
            next_indices,
            stopping_criteria.max_length,
            input_ids.view(-1, num_beams, input_ids.size(-1)).transpose(0, 1),
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        return (
            sequence_outputs["sequence_scores"],
            sequence_outputs["sequences"],
            sequence_outputs["surface_sequences"],
            (
                step_scores,
                step_input_ids,
                step_latent_input_ids,
                step_partial_surface_seq_scores,
                step_partial_latent_seq_scores,
            ),
        )

    def compute_encoder_outputs(self, input_ids, inference_batch_size):
        all_last_hidden_states = []
        encoder = self.get_encoder()
        for batch_start in range(0, input_ids.size(0), inference_batch_size):
            batch_input_ids = input_ids[
                batch_start : batch_start + inference_batch_size
            ].cuda()
            batch_attention_mask = batch_input_ids.ne(self.tokenizer.pad_token_id)
            batch_encoder_outputs = encoder(
                batch_input_ids, attention_mask=batch_attention_mask, return_dict=True,
            )
            all_last_hidden_states.append(batch_encoder_outputs.last_hidden_state.cpu())
        all_last_hidden_states = torch.cat(all_last_hidden_states, dim=0)
        # monkey patching this so we don't need to re-intialize an output object
        batch_encoder_outputs = BaseModelOutput(
            last_hidden_state=all_last_hidden_states
        )
        return batch_encoder_outputs

    @torch.no_grad()
    def fill_in_template(
        self, template, input_ids, field_dicts, encoder_outputs=None, verbose=True
    ):
        all_template_alignments = []
        list_surface_seqs = []
        if not isinstance(template, list):
            template = template.tolist()
        bar = enumerate(field_dicts)
        if verbose:
            bar = tqdm(
                bar, total=len(field_dicts), desc="filling in templates", leave=False,
            )
        for sample_i, field_dict in bar:
            sen_seq = []
            field_selection_dict = dict()
            if encoder_outputs is None:
                sample_encoder_outputs = None
            else:
                sample_encoder_outputs = (
                    encoder_outputs[sample_i].to(self.device),
                    None,
                    None,
                )
            template_alignment = []

            past_key_values = None
            for template_list_i, token_i in enumerate(template):
                template_alignment.append(len(sen_seq))
                if token_i in self.tokenizer.all_field_tokens:
                    if token_i in field_dict:
                        if token_i in self.tokenizer.field_start_set:
                            previous_selected_indices = list(
                                range(len(field_dict[token_i]))
                            )
                        else:
                            previous_selected_indices = field_selection_dict[token_i]

                        if len(previous_selected_indices) > 1:
                            # fix edge case when greedy search cannot fix this
                            previous_selected_indices = [
                                i
                                for i in previous_selected_indices
                                if field_dict[token_i][i] != -1
                            ]
                            assert len(previous_selected_indices) > 0
                            partial_seq = torch.LongTensor(sen_seq).unsqueeze(0).cuda()
                            if past_key_values is not None:
                                partial_seq = partial_seq[
                                    :, past_key_values[0][0].size(-2) :
                                ]
                            sample_input_ids = input_ids[sample_i].unsqueeze(0).cuda()

                            outputs = self(
                                input_ids=None
                                if sample_encoder_outputs
                                else sample_input_ids,
                                attention_mask=sample_input_ids.ne(
                                    self.tokenizer.pad_token_id
                                ),
                                decoder_input_ids=partial_seq,
                                use_cache=True,
                                return_dict=True,
                                encoder_outputs=sample_encoder_outputs,
                                past_key_values=past_key_values,
                            )
                            sample_encoder_outputs = (
                                outputs["encoder_last_hidden_state"],
                                None,
                                None,
                            )
                            past_key_values = outputs["past_key_values"]
                            output_logits = outputs.logits
                            last_token_prob = output_logits[0, -1].log_softmax(dim=-1)
                            candidate_scores = last_token_prob[
                                [
                                    field_dict[token_i][idx]
                                    for idx in previous_selected_indices
                                ]
                            ]
                            _, candidate_idx = candidate_scores.max(dim=0)
                            selected_token = field_dict[token_i][
                                previous_selected_indices[candidate_idx.item()]
                            ]

                            selected_indices = []
                            for j in previous_selected_indices:
                                if field_dict[token_i][j] == selected_token:
                                    selected_indices.append(j)
                            assert len(selected_indices) > 0
                            next_field_token = self.tokenizer.next_field_token_map.get(
                                token_i, None
                            )
                            if next_field_token is not None:
                                field_selection_dict[
                                    next_field_token
                                ] = selected_indices

                            sen_seq.append(selected_token)
                        else:
                            fill_in_token = field_dict[token_i][
                                previous_selected_indices[0]
                            ]
                            next_field_token = self.tokenizer.next_field_token_map.get(
                                token_i, None
                            )
                            if next_field_token is not None:
                                field_selection_dict[
                                    next_field_token
                                ] = previous_selected_indices
                            if fill_in_token == -1:
                                continue
                            sen_seq.append(fill_in_token)

                        # handling run over tokens
                        if token_i in self.tokenizer.field_cutoff_set:
                            next_run_over_token = self.tokenizer.next_field_token_map.get(
                                token_i, None
                            )
                            previous_selected_indices = field_selection_dict[token_i]
                            selected_idx = previous_selected_indices[0]
                            while (
                                next_run_over_token is not None
                                and next_run_over_token in field_dict
                            ):
                                fill_in_token = field_dict[next_run_over_token][
                                    selected_idx
                                ]
                                if fill_in_token != -1:
                                    sen_seq.append(fill_in_token)
                                    next_run_over_token = self.tokenizer.next_field_token_map.get(
                                        next_run_over_token, None
                                    )
                                else:
                                    break
                    elif token_i in self.tokenizer.field_start_set:
                        # CONDITION: not in field dict but is start of a field, we will keep it.
                        sen_seq.append(token_i)
                    else:
                        # CONDITION: not in field dict but is not start of a field, we will skip it.
                        continue
                elif token_i == self.tokenizer.pad_token_id:
                    # CONDITION: skipping pad tokens
                    continue
                else:
                    sen_seq.append(token_i)
            list_surface_seqs.append({"input_ids": sen_seq})
            template_alignment.append(len(sen_seq))
            all_template_alignments.append(template_alignment)

        padded_results = self.tokenizer.pad(list_surface_seqs)
        surface_seqs = torch.LongTensor(padded_results["input_ids"])
        surface_attention_mask = torch.LongTensor(padded_results["attention_mask"])

        return surface_seqs, surface_attention_mask, all_template_alignments

    @torch.no_grad()
    def compute_template_stats(
        self, template, input_ids, field_dicts, inference_batch_size=32
    ):
        (
            surface_seqs,
            surface_attention_mask,
            template_alignments,
        ) = self.fill_in_template(template, input_ids, field_dicts)

        logprobs = []
        for batch_start in range(0, input_ids.size(0), inference_batch_size):
            # TODO: fix hard-coded batch size and device
            batch_input_ids = input_ids[
                batch_start : batch_start + inference_batch_size
            ].cuda()
            batch_surface_seqs = surface_seqs[
                batch_start : batch_start + inference_batch_size
            ].cuda()

            batch_outputs = self(
                input_ids=batch_input_ids, decoder_input_ids=batch_surface_seqs,
            )

            batch_logprobs = (
                batch_outputs.logits[:, :-1]
                .log_softmax(dim=-1)
                .gather(index=batch_surface_seqs[:, 1:].unsqueeze(-1), dim=-1)
                .squeeze(-1)
            )
            logprobs.append(batch_logprobs.cpu())

        logprobs = torch.cat(logprobs, dim=0)
        surface_seq_scores = (logprobs * surface_attention_mask[:, 1:].float()).sum(
            dim=1
        )

        sort_indices = surface_seq_scores.sort(dim=0).indices

        return {
            "word_scores": logprobs,
            "seq_scores": surface_seq_scores,
            "sort_indices": sort_indices,
            "seq": surface_seqs,
            "alignment": template_alignments,
        }

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ):
        # init values
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(
                stopping_criteria, max_length
            )
        if len(stopping_criteria) == 0:
            warnings.warn(
                "You don't have defined any stopping_criteria, this will likely loop forever",
                UserWarning,
            )
        pad_token_id = (
            pad_token_id if pad_token_id is not None else self.config.pad_token_id
        )
        eos_token_id = (
            eos_token_id if eos_token_id is not None else self.config.eos_token_id
        )
        output_scores = (
            output_scores if output_scores is not None else self.config.output_scores
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        if "last_hidden_state" in model_kwargs["encoder_outputs"]:
            model_kwargs["encoder_outputs"]["last_hidden_state"] = model_kwargs[
                "encoder_outputs"
            ]["last_hidden_state"].to(self.device)
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        assert (
            num_beams * batch_size == batch_beam_size
        ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only

        step_beam_scores = []
        step_input_ids = []
        for _ in range(cur_len, max_length):

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len
            )
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
                next_token_scores
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(
                    model_kwargs["past"], beam_idx
                )

            step_beam_scores.append(beam_scores.cpu())
            step_input_ids.append(input_ids.cpu())

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"], (step_beam_scores, step_input_ids)

    @torch.no_grad()
    def batch_input_ids_fill_in_template(
        self,
        template,
        input_ids,
        field_dicts,
        encoder_outputs=None,
        verbose=True,
        return_log_scores=True,
        inference_batch_size=32,
    ):
        """
        Filling in a single template for multiple input_ids.
        :param template:
        :param input_ids:
        :param field_dicts:
        :param encoder_outputs:
        :param verbose:
        :return:
        """
        if not isinstance(template, list):
            template = template.tolist()

        num_inputs = len(field_dicts)
        # template[0] should always be <s>
        surface_seqs = [
            [] for _ in range(num_inputs)
        ]  # surface sequences for each input
        all_template_alignments = [[] for _ in range(num_inputs)]
        all_field_selection_dicts = [dict() for _ in range(num_inputs)]

        if encoder_outputs is None:
            encoder_outputs = self.compute_encoder_outputs(
                input_ids, inference_batch_size
            )
        else:
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)

        bar = enumerate(template)
        if verbose:
            bar = tqdm(bar, total=len(template))
        for template_list_i, token_i in bar:
            for i in range(num_inputs):
                all_template_alignments[i].append(len(surface_seqs[i]))

            if token_i not in self.tokenizer.all_field_tokens:
                for surface_seq in surface_seqs:
                    surface_seq.append(token_i)
            elif token_i == self.tokenizer.pad_token_id:
                continue
            else:
                for batch_start in range(0, num_inputs, inference_batch_size):
                    batch_decoder_input_ids = utils.pad_list(
                        surface_seqs[batch_start : batch_start + inference_batch_size],
                        self.tokenizer.pad_token_id,
                    ).cuda()
                    batch_encoder_last_state = encoder_outputs.last_hidden_state[
                        batch_start : batch_start + inference_batch_size
                    ].cuda()
                    batch_attention_mask = (
                        input_ids[batch_start : batch_start + inference_batch_size]
                        .ne(self.tokenizer.pad_token_id)
                        .cuda()
                    )
                    batch_encoder_outputs = BaseModelOutput(
                        last_hidden_state=batch_encoder_last_state
                    )
                    batch_outputs = self.forward(
                        input_ids=None,
                        attention_mask=batch_attention_mask,
                        decoder_input_ids=batch_decoder_input_ids,
                        use_cache=True,
                        return_dict=True,
                        encoder_outputs=batch_encoder_outputs,
                    )
                    batch_logprobs = batch_outputs.logits.log_softmax(dim=-1)
                    last_token_index = (
                        batch_decoder_input_ids.ne(self.tokenizer.pad_token_id).sum(
                            dim=-1
                        )
                        - 1
                    )
                    batch_next_token_scores = torch.stack(
                        [
                            batch_logprobs[i, j]
                            for i, j in enumerate(list(last_token_index))
                        ],
                        dim=0,
                    ).cpu()
                    del batch_logprobs

                    for batch_i, sample_i in enumerate(
                        range(
                            batch_start,
                            min(batch_start + inference_batch_size, num_inputs),
                        )
                    ):

                        if token_i in field_dicts[sample_i]:
                            if token_i in self.tokenizer.field_start_set:
                                previous_selected_indices = list(
                                    range(len(field_dicts[sample_i][token_i]))
                                )
                            else:
                                if token_i not in all_field_selection_dicts[sample_i]:
                                    # EDGE CASE: template fragments, skipping
                                    continue
                                previous_selected_indices = all_field_selection_dicts[
                                    sample_i
                                ][token_i]

                            if len(previous_selected_indices) > 1:
                                # fix edge case when greedy search cannot fix this
                                previous_selected_indices = [
                                    i
                                    for i in previous_selected_indices
                                    if field_dicts[sample_i][token_i][i] != -1
                                ]
                                assert len(previous_selected_indices) > 0
                                candidate_scores = batch_next_token_scores[batch_i][
                                    [
                                        field_dicts[sample_i][token_i][idx]
                                        for idx in previous_selected_indices
                                    ]
                                ]
                                _, candidate_idx = candidate_scores.max(dim=0)
                                selected_token = field_dicts[sample_i][token_i][
                                    previous_selected_indices[candidate_idx.item()]
                                ]
                                selected_indices = []
                                for j in previous_selected_indices:
                                    if (
                                        field_dicts[sample_i][token_i][j]
                                        == selected_token
                                    ):
                                        selected_indices.append(j)
                                assert len(selected_indices) > 0
                                next_field_token = self.tokenizer.next_field_token_map.get(
                                    token_i, None
                                )
                                if next_field_token is not None:
                                    all_field_selection_dicts[sample_i][
                                        next_field_token
                                    ] = selected_indices
                                surface_seqs[sample_i].append(selected_token)
                            else:
                                fill_in_token = field_dicts[sample_i][token_i][
                                    previous_selected_indices[0]
                                ]
                                next_field_token = self.tokenizer.next_field_token_map.get(
                                    token_i, None
                                )
                                if next_field_token is not None:
                                    all_field_selection_dicts[sample_i][
                                        next_field_token
                                    ] = previous_selected_indices
                                if fill_in_token == -1:
                                    continue
                                surface_seqs[sample_i].append(fill_in_token)

                            # handling run over tokens
                            if token_i in self.tokenizer.field_cutoff_set:
                                multi_runover_tokens = field_dicts[sample_i].get(
                                    (token_i, "runover"), None
                                )
                                if multi_runover_tokens is None:
                                    continue
                                assert len(multi_runover_tokens) != 0
                                previous_selected_indices = all_field_selection_dicts[
                                    sample_i
                                ][token_i]
                                selected_idx = previous_selected_indices[0]
                                for fill_in_token in multi_runover_tokens[selected_idx]:
                                    if fill_in_token != -1:
                                        surface_seqs[sample_i].append(fill_in_token)
                        elif token_i in self.tokenizer.field_start_set:
                            # CONDITION: not in field dict but is start of a field, we will keep it.
                            surface_seqs[sample_i].append(token_i)
                        else:
                            # CONDITION: not in field dict but is not start of a field, we will skip it.
                            continue

        for i in range(num_inputs):
            all_template_alignments[i].append(len(surface_seqs[i]))
        surface_seqs = utils.pad_list(surface_seqs, self.tokenizer.pad_token_id)
        surface_attention_mask = surface_seqs.eq(self.tokenizer.pad_token_id)

        if return_log_scores:
            output_logprobs = []
            for batch_start in range(0, num_inputs, inference_batch_size):
                batch_decoder_input_ids = surface_seqs[
                    batch_start : batch_start + inference_batch_size
                ].cuda()
                batch_encoder_last_state = encoder_outputs.last_hidden_state[
                    batch_start : batch_start + inference_batch_size
                ].cuda()
                batch_attention_mask = (
                    input_ids[batch_start : batch_start + inference_batch_size]
                    .ne(self.tokenizer.pad_token_id)
                    .cuda()
                )
                batch_encoder_outputs = BaseModelOutput(
                    last_hidden_state=batch_encoder_last_state
                )
                batch_outputs = self.forward(
                    input_ids=None,
                    attention_mask=batch_attention_mask,
                    decoder_input_ids=batch_decoder_input_ids,
                    use_cache=True,
                    return_dict=True,
                    encoder_outputs=batch_encoder_outputs,
                )
                batch_output_logprobs = (
                    batch_outputs.logits[:, :-1]
                    .log_softmax(dim=-1)
                    .gather(index=batch_decoder_input_ids[:, 1:].unsqueeze(-1), dim=-1)
                    .squeeze(-1)
                )
                batch_output_logprobs = batch_output_logprobs * (
                    batch_decoder_input_ids[:, 1:].ne(self.tokenizer.pad_token_id)
                )
                output_logprobs.append(batch_output_logprobs.cpu())
                del batch_output_logprobs
            output_logprobs = torch.cat(output_logprobs)
            return (
                surface_seqs,
                surface_attention_mask,
                all_template_alignments,
                output_logprobs,
            )
        else:
            return surface_seqs, surface_attention_mask, all_template_alignments


class TemplateSearchScorer(BeamSearchScorer):
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
        **kwargs,
    ):
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            TemplateHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor(
            [False for _ in range(batch_size)], dtype=torch.bool, device=self.device
        )

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if (
            not isinstance(num_beam_groups, int)
            or (num_beam_groups > num_beams)
            or (num_beams % num_beam_groups != 0)
        ):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

        if "max_length" in kwargs:
            warnings.warn(
                "Passing `max_length` to BeamSearchScorer is deprecated and has no effect."
                "`max_length` should be passed directly to `beam_search(...)`, `beam_sample(...)`"
                ",or `group_beam_search(...)`."
            )

    def process(
        self,
        latent_input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        surface_input_ids: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        cur_len = latent_input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (latent_input_ids.shape[0] // self.group_size)

        device = latent_input_ids.device
        next_beam_scores = torch.zeros(
            (batch_size, self.group_size), dtype=next_scores.dtype, device=device
        )
        next_beam_tokens = torch.zeros(
            (batch_size, self.group_size), dtype=next_tokens.dtype, device=device
        )
        next_beam_indices = torch.zeros(
            (batch_size, self.group_size), dtype=next_indices.dtype, device=device
        )

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), f"Batch can only be done if at least {self.num_beams} beams have been generated"
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(
                    next_tokens[batch_idx],
                    next_scores[batch_idx],
                    next_indices[batch_idx],
                )
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = (
                        beam_token_rank >= self.group_size
                    )
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        latent_input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                        surface_input_ids[batch_beam_idx].clone(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        latent_input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        surface_input_ids: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = latent_input_ids[batch_beam_idx]
                final_surface_tokens = surface_input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score, final_surface_tokens)

        # select the best hypotheses
        sent_lengths = latent_input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(
            batch_size * self.num_beam_hyps_to_keep,
            device=self.device,
            dtype=torch.float32,
        )
        best_surface_input_ids = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score
                best_surface_input_ids.append(best_hyp_tuple[2])

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded: torch.LongTensor = latent_input_ids.new(
            batch_size * self.num_beam_hyps_to_keep, sent_max_len
        )
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "surface_sequences": best_surface_input_ids,
            }
        )


class TemplateHypotheses(BeamHypotheses):
    def add(
        self,
        latent_input_ids: torch.LongTensor,
        sum_logprobs: float,
        surface_input_ids: torch.LongTensor,
    ):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (latent_input_ids.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, latent_input_ids, surface_input_ids))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted(
                    [(s, idx) for idx, (s, _, _) in enumerate(self.beams)]
                )
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
