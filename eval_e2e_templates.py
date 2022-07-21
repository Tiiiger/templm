import os
import torch
from pathlib import Path

from transformers import (
    BartForConditionalGeneration,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)
from train_seq2seq import ModelArguments, DataTrainingArguments
from template_args import TemplateArguments
import utils
import load_utils
from data import TabularData
from tqdm import tqdm, trange
import numpy as np
from field_transformation import _e2e_field_transformations_complex

parser = HfArgumentParser(
    (
        ModelArguments,
        DataTrainingArguments,
        Seq2SeqTrainingArguments,
        TemplateArguments,
    )
)

(
    model_args,
    data_args,
    training_args,
    template_args,
) = parser.parse_args_into_dataclasses()

model, tokenizer, no_space_tokenizer = load_utils.load_model_and_tokenizer(model_args)
model.init_for_template_search(tokenizer, None, False)
model.resize_token_embeddings(len(tokenizer))
model.cuda()

train_dataset = TabularData(
    data_args.dataset_name,
    "train",
    tokenizer,
    no_space_tokenizer,
    data_args,
    training_args,
    template_args.field_tokens_cutoff,
)

# reload tokenizer because special tokens might be different across loading
# TODO: figure out why
tokenizer = AutoTokenizer.from_pretrained(training_args.output_dir)

eval_dataset = TabularData(
    data_args.dataset_name,
    "validation" if template_args.evaluation_split == "val" else "test",
    tokenizer,
    no_space_tokenizer,
    data_args,
    training_args,
    template_args.field_tokens_cutoff,
)

# Load Template
template_dir = Path(training_args.output_dir)

all_templates_temp, type_combination_indices = utils.load_templates(
    template_dir, template_args
)
from collections import defaultdict
import re

all_templates = defaultdict(list)
for k, v in all_templates_temp.items():
    for template in v:
        template_str = utils.decode_and_clean_template(
            template, tokenizer, train_dataset.field_max_lens
        )
        field_combination = tuple(
            sorted([f[1:-1] for f in re.findall(r"\[[a-zA-Z]+\]", template_str)])
        )  # detect all the tags that look like <{some characters}_{digit}>
        all_templates[field_combination].append(template)
        k = tuple(f.upper() for f in k)
        all_templates[k].append(template)

# Evaluate Lexicalization and Validity
all_lexicalizations = set()
for v in _e2e_field_transformations_complex.values():
    all_lexicalizations.update(v)
for v in train_dataset.field_possible_values.values():
    all_lexicalizations.update(v)

lex_templates = set()
lex_in_templates = []
invalid_templates = set()
unique_template_strs = set()
for k, v in all_templates.items():
    templates = []
    seen_template_str = set()
    for template in v:
        template_str = utils.decode_and_clean_template(
            template, tokenizer, train_dataset.field_max_lens
        )
        unique_template_strs.add(template_str)
        if template_str in seen_template_str:
            continue
        is_clean = True
        if "<" in template_str:
            invalid_templates.add(template_str)
            is_clean = False
        for lex in all_lexicalizations:
            if lex in template_str and lex != "restaurant":
                is_clean = False
                lex_templates.add(template_str)
                lex_in_templates.append(lex)
        if template_args.eval_postprocess and is_clean:
            templates.append(template)
        elif not template_args.eval_postprocess:
            templates.append(template)
        seen_template_str.add(template_str)
    all_templates[k] = templates
template_count = len(unique_template_strs)
print(
    f"Out of {template_count} unique templates",
    f" {len(lex_templates)} {len(lex_templates) / template_count:.2%} contain lexicalization",
    f" {len(invalid_templates)} {len(invalid_templates) / template_count:.2%} contain invalid field",
)

data_collator = utils.get_datacollator(tokenizer, model, data_args, training_args)

best_outputs = []
all_input_ids = []
all_sen_ids = []
all_sen_scores = []
all_templates_expanded = []
boundaries = [0]
count = 0

from itertools import combinations

for i, sample in tqdm(
    enumerate(eval_dataset.processed_dataset),
    total=len(eval_dataset.processed_dataset),
):
    sample_field_combination = tuple(
        [f.upper() for f in eval_dataset.indices_to_combination[i]]
    )
    if sample_field_combination in all_templates:
        type_templates = all_templates[sample_field_combination]
    else:
        print("unseen: ", sample_field_combination)
        type_templates = []
        for backoff_len in range(1, len(sample_field_combination)):
            for backoff_field_combination in combinations(
                sample_field_combination, backoff_len
            ):
                if backoff_field_combination in all_templates:
                    type_templates.extend(all_templates[backoff_field_combination])
        if len(type_templates) == 0:
            print("failed to backoff. searching over all templates")
            for templates in all_templates.values():
                type_templates.extend(templates)

    if template_args.random_selection_inference:
        import random
        type_template_str = [utils.decode_and_clean_template(
            t, tokenizer, train_dataset.field_max_lens
        ) for t in type_templates]
        num_fields = [len(tuple(
            sorted([f[1:-1] for f in re.findall(r"\[[a-zA-Z]+\]", t_str)])
        )) for t_str in type_template_str]
        num_max_fields = max(num_fields)
        idx_with_max_num_fields = [i for i, n in enumerate(num_fields) if n == num_max_fields]
        rand_template_i = random.choice(idx_with_max_num_fields)
        type_templates = [type_templates[rand_template_i]]

    eval_input_ids = [
        torch.LongTensor(eval_dataset.processed_dataset[i]["input_ids"])
    ]
    eval_field_dicts = [eval_dataset.field_dicts[i]]
    fill_in_sens = [
        model.fill_in_template(t, eval_input_ids, eval_field_dicts, verbose=False)[
            0
        ]
        .squeeze(0)
        .tolist()
        for t in type_templates
    ]
    all_templates_expanded.extend(type_templates)
    all_sen_ids.extend(fill_in_sens)
    all_input_ids.extend(
        [
            eval_dataset.processed_dataset[i]["input_ids"]
            for _ in range(len(fill_in_sens))
        ]
    )
    count += len(fill_in_sens)
    boundaries.append(count)

for batch_start in trange(
    0,
    len(all_sen_ids),
    template_args.inference_batch_size,
    desc="computing template scores",
):
    batch_output_ids = utils.pad_list(
        all_sen_ids[batch_start : batch_start + template_args.inference_batch_size],
        tokenizer.pad_token_id,
    ).cuda()
    batch_input_ids = utils.pad_list(
        all_input_ids[
            batch_start : batch_start + template_args.inference_batch_size
        ],
        tokenizer.pad_token_id,
    ).cuda()
    output_logits = model(
        input_ids=batch_input_ids, decoder_input_ids=batch_output_ids
    ).logits
    output_logprobs = (
        output_logits[:, :-1]
        .log_softmax(dim=-1)
        .gather(index=batch_output_ids[:, 1:].unsqueeze(-1), dim=-1)
        .squeeze(-1)
    )
    output_logprobs = output_logprobs * (
        batch_output_ids[:, 1:].ne(tokenizer.pad_token_id)
    )
    output_logprobs = output_logprobs.sum(dim=-1)
    if template_args.length_penalty > 0:
        num_non_pad = batch_output_ids[:, 1:].ne(tokenizer.pad_token_id).sum(dim=-1)
        output_logprobs = output_logprobs / num_non_pad.float().pow(
            template_args.length_penalty
        )
    all_sen_scores.extend(output_logprobs.tolist())

return_sens = []
return_templates = []
for i in range(len(eval_dataset.processed_dataset)):
    sample_scores = all_sen_scores[boundaries[i] : boundaries[i + 1]]
    best_template_i = np.argmax(sample_scores)
    return_sens.append(
        tokenizer.decode(
            all_sen_ids[boundaries[i] + best_template_i], skip_special_tokens=True
        )
    )
    return_templates.append(
        utils.decode_and_clean_template(
            all_templates_expanded[boundaries[i] + best_template_i],
            tokenizer,
            train_dataset.field_max_lens,
        )
    )

if template_args.local_search_prune_topk is not None:
    template_dir = template_dir / f"prune{template_args.local_search_prune_topk}"

if template_args.random_selection_inference:
    template_dir = template_dir / "random_inference"


if template_args.evaluation_split == "val":
    split_name =  "validation"
    output_dir = template_dir / "dev_output"
elif template_args.evaluation_split == "test":
    split_name = "test"
    output_dir = template_dir / "test_output"

os.makedirs(output_dir, exist_ok=True)

with open(output_dir / "lex_stats.txt", "w") as f:
    f.write(f"#. Templates: {template_count}\n")
    f.write(f"#. Lex. Templates: {len(lex_templates)}\n")
    f.write(f"%. Lex. Templates: {len(lex_templates)/template_count:.2%}\n")

file_name = (
    "typed_best_outputs_postprocessed.txt"
    if template_args.eval_postprocess
    else "typed_best_outputs.txt"
)
with open(output_dir / file_name, "w") as f:
    for t in return_sens:
        f.write(t.strip() + "\n")
file_name = (
    "typed_best_templates_postprocessed.txt"
    if template_args.eval_postprocess
    else "typed_best_templates.txt"
)
with open(output_dir / file_name, "w") as f:
    for t in return_templates:
        f.write(t.strip() + "\n")

file_name = "input_data.txt"
with open(output_dir / file_name, "w") as f:
    for t in eval_dataset.raw_datasets[split_name]:
        f.write(t["meaning_representation"].strip() + "\n")
