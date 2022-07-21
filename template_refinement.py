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
from refine_utils import *
from load_utils import *
from copy import deepcopy

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
    print("loading AR model.")
    model, tokenizer, no_space_tokenizer = load_model_and_tokenizer(model_args)
    print("loading Infilling model.")
    model_args.model_name_or_path = template_args.refinement_infill_model_path
    infill_model, _, _ = load_model_and_tokenizer(
        model_args, load_tokenizer=False, load_no_space_tokenizer=False
    )

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
            template_args.field_tokens_cutoff,
        )
    if data_args.data_per_type != -1:
        tabular_data.few_shot_sample(data_args.data_per_type)
    # Handle this in the model
    model.resize_token_embeddings(len(tokenizer))
    model.init_for_template_search(
        tokenizer, tabular_data.field_dicts, template_args.hallucinate_aug
    )
    infill_model.resize_token_embeddings(len(tokenizer))
    infill_model.init_for_template_search(
        tokenizer, tabular_data.field_dicts, template_args.hallucinate_aug
    )
    infill_model.config.forced_bos_token_id = None

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
    infill_model.cuda()

    for type_i in trange(num_types):
        input_type_dir = Path(template_args.refinement_template_dir) / f"type_{type_i}"
        exact_templates = torch.load(input_type_dir / "templates.pt")
        chunk_indices = torch.load(input_type_dir / "chunked_indices.pt")
        chunk_probs = torch.load(input_type_dir / "chunked_probs.pt")
        alignments = torch.load(input_type_dir / "alignments.pt")
        fill_in_ids = torch.load(input_type_dir / "fill_in_ids.pt")
        if "e2e" in data_args.dataset_name:
            field_combination = torch.load(input_type_dir / "field_combination.pt")

        # for type_i in trange(num_types):
        type_dir = Path(training_args.output_dir, f"type_{type_i}")
        if (
            os.path.isfile(type_dir / "templates.pt")
            and not training_args.overwrite_output_dir
        ):
            print(f"Type {type_i} done: skipping")
            continue
        type_dir.mkdir(parents=True, exist_ok=True)
        if "synthbio" in data_args.dataset_name:
            if template_args.local_search_augment > 0:
                raise NotImplementedError
            type_indices = list(notable_type_to_indices.values())[type_i]
            type_input_ids = [
                tabular_data.processed_dataset[i]["input_ids"] for i in type_indices
            ]
            type_field_dicts = [tabular_data.field_dicts[i] for i in type_indices]

            type_input_ids = pad_list(type_input_ids, tokenizer.pad_token_id)
            type_input_ids = torch.LongTensor(type_input_ids)
        elif "e2e" in data_args.dataset_name:
            type_input_ids = torch.load(input_type_dir / "type_input_ids.pt")
            type_field_dicts = torch.load(input_type_dir / "type_field_dicts.pt")
        else:
            raise NotImplemented

        avg_log_score = []
        for i in range(len(exact_templates)):
            len_norm_sen_scores = chunk_probs[i].sum(dim=1) / fill_in_ids[i].ne(
                tokenizer.pad_token_id
            ).sum(dim=1)
            avg_log_score.append(len_norm_sen_scores.mean().item())

        return_templates = []
        refined_templates = []
        refined_const_input = []
        refined_const_output = []
        template_sort_ids = np.argsort(avg_log_score)[::-1]
        for i in trange(template_args.refinement_topn_delex):
            if i >= len(template_sort_ids):
                continue
            template_i = template_sort_ids[i]
            print(
                decode_and_clean_template(
                    exact_templates[template_i], tokenizer, tabular_data.field_max_lens,
                )
            )
            # valid constituents that got matched with scores
            valid_consts = sort_valid_consts(
                tokenizer,
                exact_templates[template_i],
                fill_in_ids[template_i][template_i],
                alignments[template_i][template_i],
                chunk_indices[template_i],
                alignments[template_i],
                fill_in_ids[template_i],
                chunk_probs[template_i],
                lexicalized_only=template_args.refinement_lex_only,
            )
            filtered_consts = []
            for c in valid_consts:
                if c[-2] < template_args.refinement_const_cutoff:
                    filtered_consts.append(c)
            sorted_consts = sorted(filtered_consts, key=lambda x: len(x[-1]))

            to_refine_consts = []
            refined_chunks = set()
            for c in sorted_consts[::-1]:
                child_or_partent_refined = False
                for to_refine_const in to_refine_consts:
                    if span_contains(c, to_refine_const):
                        child_or_partent_refined = True
                if child_or_partent_refined or tuple(c[-1]) in refined_chunks:
                    continue
                else:
                    to_refine_consts.append(c)
                    refined_chunks.add(tuple(c[-1]))
            to_refine_consts = list(sorted(to_refine_consts, key=lambda x: x[2]))
            for to_refine_const in to_refine_consts:
                print("Constituent to refine: ", to_refine_const)

            chunk_indices_predit = deepcopy(chunk_indices)

            sample_refinement_templates = []
            sample_refinement_inputs = []
            sample_refinement_outputs = []
            for loop_i, to_refine_const in enumerate(to_refine_consts):
                infill_input_ids = []

                infill_start_align_i = chunk_indices_predit[template_i][
                    to_refine_const[-1][0]
                ]
                infill_end_align_i = chunk_indices_predit[template_i][
                    to_refine_const[-1][-1] + 1
                ]

                mask_tokens = [
                    [tokenizer.mask_token_id]
                    for _ in range(len(fill_in_ids[template_i]))
                ]
                masked_fill_in_ids = infill_tokens(
                    fill_in_ids[template_i],
                    mask_tokens,
                    alignments[template_i],
                    infill_start_align_i,
                    infill_end_align_i,
                    tokenizer.pad_token_id,
                )
                for sample_i in range(len(alignments[template_i])):
                    type_input_ids_list = type_input_ids[sample_i][
                        : type_input_ids[sample_i]
                        .ne(tokenizer.pad_token_id)
                        .sum(dim=-1)
                    ].tolist()
                    infill_input_ids.append(
                        type_input_ids_list + masked_fill_in_ids[sample_i][1:]
                    )
                infill_input_ids = pad_list(infill_input_ids, tokenizer.pad_token_id)

                (
                    refine_cand_scores,
                    refine_cands,
                    refine_surface_seqs,
                    _,
                ) = infill_model.generate(
                    infill_input_ids,
                    field_dicts=type_field_dicts,
                    inference_batch_size=template_args.inference_batch_size,
                    input_weighting=torch.full(
                        (infill_input_ids.size(0),),
                        fill_value=1 / infill_input_ids.size(0),
                    ),
                    search_crit=template_args.search_crit,
                    recompute_log_prob=False,
                    max_length=template_args.max_decode_steps,
                    tokenizer=tokenizer,
                    num_beams=data_args.num_beams,
                    num_return_sequences=data_args.num_beams,
                    length_penalty=template_args.length_penalty,
                    early_stopping=True,
                    verbose=False,
                    mode="template_search",
                )

                refine_cand = refine_cands[0]
                refine_cand = refine_cand[
                    1 : refine_cand.ne(tokenizer.pad_token_id).sum() - 1
                ].tolist()
                template_refinment_start = chunk_indices[template_i][
                    to_refine_const[-1][0]
                ]
                template_refinment_end = chunk_indices[template_i][
                    to_refine_const[-1][-1] + 1
                ]
                sample_refinement_input = exact_templates[template_i][
                    template_refinment_start:template_refinment_end
                ]
                sample_refinement_inputs.append(sample_refinement_input)
                sample_refinement_outputs.append(refine_cand)
                print(
                    tokenizer.decode(sample_refinement_input),
                    "-> <delete>"
                    if len(refine_cand) == 0
                    else f"-> {tokenizer.decode(refine_cand)}",
                )

                num_template_tokens_diff = len(refine_cand) - (
                    template_refinment_end - template_refinment_start
                )

                chunk_end = to_refine_const[-1][-1]
                for chunk_i in range(chunk_end, len(chunk_indices[template_i])):
                    chunk_indices[template_i][chunk_i] += num_template_tokens_diff

                exact_templates[template_i] = (
                    exact_templates[template_i][:template_refinment_start]
                    + refine_cand
                    + exact_templates[template_i][template_refinment_end:]
                )
                sample_refinement_templates.append(exact_templates[template_i])
                print(
                    decode_and_clean_template(
                        exact_templates[template_i],
                        tokenizer,
                        tabular_data.field_max_lens,
                    )
                )

                refine_surface_ids = [
                    refine_surface_seqs[0][sample_i][
                        1 : refine_surface_seqs[0][sample_i]
                        .ne(tokenizer.pad_token_id)
                        .sum()
                    ].tolist()
                    for sample_i in range(len(alignments[template_i]))
                ]

                new_fill_in_ids = pad_list(
                    infill_tokens(
                        fill_in_ids[template_i],
                        refine_surface_ids,
                        alignments[template_i],
                        infill_start_align_i,
                        infill_end_align_i,
                        tokenizer.pad_token_id,
                    ),
                    tokenizer.pad_token_id,
                )

                for sample_i in range(len(alignments[template_i])):
                    num_tokens_diff = len(refine_surface_ids[sample_i]) - (
                        alignments[template_i][sample_i][infill_end_align_i]
                        - alignments[template_i][sample_i][infill_start_align_i]
                    )

                    align_end = chunk_indices_predit[template_i][
                        to_refine_const[-1][-1]
                    ]
                    for align_i in range(
                        align_end, len(alignments[template_i][sample_i])
                    ):
                        alignments[template_i][sample_i][align_i] += num_tokens_diff

                fill_in_ids[template_i] = new_fill_in_ids
            return_templates.append(exact_templates[template_i])
            refined_templates.append(sample_refinement_templates)
            refined_const_input.append(sample_refinement_inputs)
            refined_const_output.append(sample_refinement_outputs)

        if (
            not os.path.isfile(type_dir / "template.pt")
            or training_args.overwrite_output_dir
        ):
            if "e2e" in data_args.dataset_name:
                torch.save(
                    field_combination,
                    type_dir / "field_combination.pt",
                )
            torch.save(return_templates, type_dir / "templates.pt")
            torch.save(refined_templates, type_dir / "refined_templates.pt")
            torch.save(refined_const_input, type_dir / "refine_inputs.pt")
            torch.save(refined_const_output, type_dir / "refine_outputs.pt")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
