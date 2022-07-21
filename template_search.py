#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys

import datasets
from data import TabularData

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from train_seq2seq import ModelArguments, DataTrainingArguments
from template_args import TemplateArguments
import torch
from pathlib import Path
from template_search_bart import TemplateSearchBART
from transformers import BartForConditionalGeneration
from tqdm import trange, tqdm
from utils import *
from load_utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.10.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/summarization/requirements.txt",
)

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    torch.set_grad_enabled(False)

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            Seq2SeqTrainingArguments,
            TemplateArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, template_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            template_args,
        ) = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    os.makedirs(training_args.output_dir, exist_ok=True)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # CHANGE: minimally changes the model class to be the custom class defined above
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model, tokenizer, no_space_tokenizer = load_model_and_tokenizer(model_args)

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )
    tabular_data = TabularData(
        data_args.dataset_name,
        "train",
        tokenizer,
        no_space_tokenizer,
        data_args,
        training_args,
        template_args.field_tokens_cutoff,
    )
    if template_args.local_search_augment_type:
        val_data = TabularData(
            data_args.dataset_name,
            "validation",
            tokenizer,
            no_space_tokenizer,
            data_args,
            training_args,
        )
    if data_args.data_per_type != -1:
        tabular_data.few_shot_sample(data_args.data_per_type)
    # Handle this in the model
    model.resize_token_embeddings(len(tokenizer))
    model.init_for_template_search(
        tokenizer, tabular_data.field_dicts, template_args.hallucinate_aug
    )

    # Data collator
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    from collections import defaultdict

    if "synthbio" in data_args.dataset_name:
        from collections import defaultdict

        notable_type_to_indices = defaultdict(list)
        for i, dt in enumerate(tabular_data.raw_datasets["train"]):
            input_table = dict(
                zip(
                    dt["input_text"]["table"]["column_header"],
                    dt["input_text"]["table"]["content"],
                )
            )
            notable_type_to_indices[input_table["notable_type"]].append(i)
        num_types = len(notable_type_to_indices)
    elif "e2e" in data_args.dataset_name:
        frequent_field_combination = [
            x[0]
            for x in sorted(
                tabular_data.combination_to_indices.items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )
        ]
        if template_args.local_search_augment_type:
            assert template_args.local_search_augment > 0
            for x in val_data.combination_to_indices.keys():
                if x not in frequent_field_combination:
                    print(f"Augmenting Type: adding {x}")
                    frequent_field_combination.append(x)
        num_types = len(frequent_field_combination)
    else:
        raise NotImplemented

    # sample indices
    model.cuda()

    for type_i in trange(num_types):
        type_dir = Path(training_args.output_dir, f"type_{type_i}")
        if os.path.isfile(type_dir / "template.pt"):
            continue
        type_dir.mkdir(parents=True, exist_ok=True)
        if "synthbio" in data_args.dataset_name:
            if template_args.local_search_augment > 0:
                raise NotImplementedError
            type_indices = list(notable_type_to_indices.values())[type_i]
            type_input_ids = [
                tabular_data.processed_dataset[i]["input_ids"] for i in type_indices
            ]
            type_labels = [
                tabular_data.processed_dataset[i]["labels"] for i in type_indices
            ]
            type_field_dicts = [tabular_data.field_dicts[i] for i in type_indices]
        elif "e2e" in data_args.dataset_name:
            field_combination = frequent_field_combination[type_i]
            if template_args.local_search_augment > 0:
                assert template_args.local_search_baseline
                (
                    sample_input_ids,
                    sample_field_dicts,
                    sample_raw_field_dicts,
                ) = tabular_data.sample_field_combination(
                    field_combination, template_args.local_search_augment
                )
                type_input_ids = sample_input_ids
                type_field_dicts = sample_field_dicts
            else:
                type_indices = tabular_data.combination_to_indices[field_combination]
                type_input_ids = [
                    tabular_data.processed_dataset[i]["input_ids"] for i in type_indices
                ]
                type_field_dicts = [tabular_data.field_dicts[i] for i in type_indices]
                type_labels = [
                    tabular_data.processed_dataset[i]["labels"] for i in type_indices
                ]
        else:
            raise NotImplemented

        type_input_ids = pad_list(type_input_ids, tokenizer.pad_token_id)
        type_input_ids = torch.LongTensor(type_input_ids)

        if template_args.local_search_baseline:
            type_baseline_ids = []
            for batch_start in trange(
                0, len(type_field_dicts), template_args.inference_batch_size,
            ):
                batch_input_ids = type_input_ids[
                    batch_start : batch_start + template_args.inference_batch_size
                ].cuda()
                (output_ids, _,) = model.generate(
                    batch_input_ids,
                    field_dicts=None,
                    inference_batch_size=template_args.inference_batch_size,
                    input_weighting=torch.full(
                        (batch_input_ids.size(0),),
                        fill_value=1 / batch_input_ids.size(0),
                    ),
                    search_crit=template_args.search_crit,
                    recompute_log_prob=False,
                    max_length=template_args.max_decode_steps,
                    tokenizer=tokenizer,
                    num_beams=data_args.num_beams,
                    num_return_sequences=1,
                    length_penalty=template_args.length_penalty,
                    early_stopping=False,
                    mode="beam_search",
                )
                for i in range(output_ids.size(0)):
                    num_non_pad = output_ids[i].ne(tokenizer.pad_token_id).sum()
                    type_baseline_ids.append(output_ids[i][:num_non_pad].cpu())

        type_last_hidden_states = []

        for batch_start in range(
            0, type_input_ids.size(0), template_args.inference_batch_size
        ):
            batch_input_ids = type_input_ids[
                batch_start : batch_start + template_args.inference_batch_size
            ].cuda()
            batch_attention_mask = batch_input_ids.ne(tokenizer.pad_token_id)
            batch_encoder_outputs = model.get_encoder()(
                batch_input_ids, attention_mask=batch_attention_mask, return_dict=True,
            )
            type_last_hidden_states.append(
                batch_encoder_outputs.last_hidden_state.cpu()
            )
        type_last_hidden_states = torch.cat(type_last_hidden_states, dim=0)

        all_template_strs = set()
        all_exact_templates = []
        all_fill_in_ids = []
        all_template_token_scores = []
        all_template_alignments = []
        all_aligned_probs = []
        all_chunked_probs = []
        all_chunked_indices = []

        if template_args.local_search_baseline:
            num_search = len(type_baseline_ids)
        else:
            num_search = len(type_labels)

        for idx in trange(num_search, leave=False):
            if not template_args.local_search_baseline:
                target_text = type_labels[idx][1:-1]
                raw_field_dict = tabular_data.all_transformed_data[type_indices[idx]]
            else:
                target_text = type_baseline_ids[idx][2:-1].tolist()
                if template_args.local_search_augment == 0:
                    raw_field_dict = tabular_data.all_transformed_data[
                        type_indices[idx]
                    ]
                else:
                    raw_field_dict = sample_raw_field_dicts[idx]

            if "e2e" in data_args.dataset_name:
                mode = "e2e"
            elif "synthbio" in data_args.dataset_name:
                mode = "synthbio"
            else:
                raise NotImplemented

            exact_template = exact_match_template(
                target_text,
                raw_field_dict,
                tabular_data.field_max_lens,
                tokenizer,
                no_space_tokenizer,
                mode=mode,
            )

            template_str = decode_and_clean_template(
                exact_template, tokenizer, tabular_data.field_max_lens
            )
            if template_str not in all_template_strs:
                all_template_strs.add(template_str)
            else:
                continue

            if not template_args.local_search_compute_stats:
                all_exact_templates.append(exact_template)
            else:
                (
                    output_ids,
                    attention_mask,
                    template_alignments,
                    output_logprobs,
                ) = model.batch_input_ids_fill_in_template(
                    exact_template,
                    type_input_ids,
                    type_field_dicts,
                    encoder_outputs=type_last_hidden_states,
                    return_log_scores=True,
                    verbose=False,
                    inference_batch_size=template_args.inference_batch_size,
                )
                all_fill_in_ids.append(output_ids)
                all_template_token_scores.append(output_logprobs)
                all_exact_templates.append(exact_template)
                all_template_alignments.append(template_alignments)

                (
                    aligned_logprobs,
                    chunked_logprobs,
                    chunk_indices,
                ) = align_and_chunk_logprobs(
                    exact_template, tokenizer, template_alignments, output_logprobs
                )

                all_aligned_probs.append(aligned_logprobs)
                all_chunked_probs.append(chunked_logprobs)
                all_chunked_indices.append(chunk_indices)

        if (
            not os.path.isfile(type_dir / "template.pt")
            or training_args.overwrite_output_dir
        ):
            if "e2e" in data_args.dataset_name:
                torch.save(field_combination, type_dir / "field_combination.pt")
                torch.save(type_input_ids, type_dir / "type_input_ids.pt")
                torch.save(type_field_dicts, type_dir / "type_field_dicts.pt")
            torch.save(all_exact_templates, type_dir / "templates.pt")
            torch.save(all_fill_in_ids, type_dir / "fill_in_ids.pt")
            torch.save(all_template_token_scores, type_dir / "token_scores.pt")
            torch.save(all_template_alignments, type_dir / "alignments.pt")
            torch.save(all_aligned_probs, type_dir / "aligned_probs.pt")
            torch.save(all_chunked_probs, type_dir / "chunked_probs.pt")
            torch.save(all_chunked_indices, type_dir / "chunked_indices.pt")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
