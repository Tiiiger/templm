import os
import torch
from pathlib import Path

from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)
from train_seq2seq import ModelArguments, DataTrainingArguments
from template_args import TemplateArguments
from data import TabularData
from tqdm import tqdm, trange
from utils import *
from load_utils import *

from sacrebleu import BLEU
from bert_score import BERTScorer
from rouge_score import rouge_scorer as _rouge_scorer
from preprocessing import untokenize
from collections import defaultdict

torch.set_grad_enabled(False)

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


# Load Template
template_dir = Path(training_args.output_dir)
all_templates, _ = load_templates(template_dir, template_args, mode="synthbio")

if template_args.local_search_prune_topk is not None:
    template_dir = template_dir / f"prune{template_args.local_search_prune_topk}"
if template_args.random_selection_inference:
    template_dir = template_dir / "random_inference"

os.makedirs(template_dir / "dev_output", exist_ok=True)


model, tokenizer, no_space_tokenizer = load_model_and_tokenizer(model_args)
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
)
eval_dataset = TabularData(
    data_args.dataset_name,
    template_args.evaluation_split,
    tokenizer,
    no_space_tokenizer,
    data_args,
    training_args,
)

data_collator = get_datacollator(tokenizer, model, data_args, training_args)
best_outputs = []
all_input_ids = []
all_sen_ids = []
all_sen_scores = []
all_templates_expanded = []
boundaries = [0]
count = 0

from itertools import combinations

from collections import defaultdict

notable_type_to_indices = defaultdict(list)
for i, dt in enumerate(eval_dataset.raw_datasets[template_args.evaluation_split]):
    input_table = dict(
        zip(
            dt["input_text"]["table"]["column_header"],
            dt["input_text"]["table"]["content"],
        )
    )
    notable_type_to_indices[input_table["notable_type"]].append(i)
num_types = len(notable_type_to_indices)

all_return_sens = []
all_return_ids = []
all_return_templates = []

if template_args.eval_specific_type > -1:
    type_list = [template_args.eval_specific_type]
else:
    type_list = list(range(num_types))
for type_i in type_list:
    type_indices = list(notable_type_to_indices.values())[type_i]
    type_input_ids = [
        eval_dataset.processed_dataset[i]["input_ids"] for i in type_indices
    ]
    type_input_ids = pad_list(type_input_ids, tokenizer.pad_token_id)
    type_input_ids = torch.LongTensor(type_input_ids)
    type_labels = [eval_dataset.processed_dataset[i]["labels"] for i in type_indices]
    type_field_dicts = [eval_dataset.field_dicts[i] for i in type_indices]

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
        type_last_hidden_states.append(batch_encoder_outputs.last_hidden_state.cpu())
    type_last_hidden_states = torch.cat(type_last_hidden_states, dim=0)

    type_templates = all_templates[type_i]
    type_templates = [t for t in type_templates if len(tokenizer.decode(t)) != 0]
    type_template_strs = [
        decode_and_clean_template(t, tokenizer, train_dataset.field_max_lens)
        for t in type_templates
    ]
    type_template_field_combinations = [
        set(f[1:-1].lower() for f in re.findall(r"\[[^ ]*\]", t_str))
        for t_str in type_template_strs
    ]

    all_template_output_scores = []
    all_template_output_ids = []
    for template in tqdm(type_templates):
        (
            output_ids,
            attention_mask,
            template_alignments,
            output_logprobs,
        ) = model.batch_input_ids_fill_in_template(
            template,
            type_input_ids,
            type_field_dicts,
            encoder_outputs=type_last_hidden_states,
            return_log_scores=True,
            verbose=False,
            inference_batch_size=template_args.inference_batch_size,
        )
        output_logprobs = output_logprobs.sum(dim=-1)
        if template_args.length_penalty > 0:
            num_non_pad = (
                output_ids[:, 1:]
                .ne(tokenizer.pad_token_id)
                .sum(dim=-1)
                .to(output_logprobs.device)
            )
            output_logprobs = output_logprobs / num_non_pad.float().pow(
                template_args.length_penalty
            )
        all_template_output_scores.append(output_logprobs.cpu())
        all_template_output_ids.append(output_ids.cpu())
    all_template_output_scores = torch.stack(all_template_output_scores, dim=0)
    best_template_ids = all_template_output_scores.max(dim=0)[1].tolist()
    best_template_ids = []
    for sample_i in range(len(type_field_dicts)):
        sample_field_combination = set(
            eval_dataset.all_transformed_data[type_indices[sample_i]].keys()
        )
        valid_template_i = [
            i
            for i in range(len(type_templates))
            if type_template_field_combinations[i].issubset(sample_field_combination)
        ]
        if len(valid_template_i) == 0:
            valid_template_i = list(range(len(type_templates)))

        if template_args.random_selection_inference:
            import utils
            import random

            valid_templates = [type_templates[i] for i in valid_template_i]
            valid_template_str = [utils.decode_and_clean_template(
                t, tokenizer, train_dataset.field_max_lens
            ) for t in valid_templates]
            num_fields = [len(tuple(
                sorted([f[1:-1] for f in re.findall(r"\[[a-zA-Z]+\]", t_str)])
            )) for t_str in valid_template_str]
            num_max_fields = max(num_fields)
            idx_with_max_num_fields = [i for i, n in enumerate(num_fields) if n == num_max_fields]
            rand_template_i = random.choice(idx_with_max_num_fields)
            best_template_ids.append(valid_template_i[rand_template_i])
        else:
            best_template_ids.append(
                valid_template_i[
                    all_template_output_scores[valid_template_i, sample_i].max(dim=0)[1]
                ]
            )
    type_return_sens = []
    type_return_ids = []
    for i, template_i in enumerate(best_template_ids):
        return_ids = all_template_output_ids[template_i][i]
        num_non_pad = return_ids.ne(tokenizer.pad_token_id).sum()
        return_ids = return_ids[2 : num_non_pad - 1]
        type_return_ids.append(return_ids)
        type_return_sens.append(tokenizer.decode(return_ids, skip_special_tokens=False))
    type_return_templates = [
        decode_and_clean_template(
            type_templates[template_i], tokenizer, train_dataset.field_max_lens
        )
        for i, template_i in enumerate(best_template_ids)
    ]

    if template_args.evaluation_split == "val":
        type_output_dir = template_dir / "dev_output" / f"type_{type_i}"
    else:
        type_output_dir = template_dir / "test_output" / f"type_{type_i}"
    type_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        type_return_sens, type_output_dir / "all_return_sens.pt",
    )
    torch.save(
        type_return_ids, type_output_dir / "all_return_ids.pt",
    )
    torch.save(
        type_return_templates, type_output_dir / "all_return_templates.pt",
    )

# appending the code for generating BLEU scores
data_args.dedup_input = False
eval_dataset = TabularData(
    data_args.dataset_name,
    template_args.evaluation_split,
    tokenizer,
    no_space_tokenizer,
    data_args,
    training_args,
)

# To create multiple references, we deduplicate here to get the relevant information
# dedup by input_ids
dedup_indices = []
seen_input_data = set()
dedup_map = dict()
for i, dt in enumerate(eval_dataset.raw_datasets[template_args.evaluation_split]):
    if dt["input_text"]["context"] not in seen_input_data:
        dedup_indices.append(i)
        dedup_map[dt["input_text"]["context"]] = [i]
        seen_input_data.add(dt["input_text"]["context"])
    else:
        dedup_map[dt["input_text"]["context"]].append(i)

deduped_dataset = eval_dataset.raw_datasets[template_args.evaluation_split].select(dedup_indices)
notable_type_to_indices = defaultdict(list)
for i, dt in enumerate(deduped_dataset):
    input_table = dict(
        zip(
            dt["input_text"]["table"]["column_header"],
            dt["input_text"]["table"]["content"],
        )
    )
    notable_type_to_indices[input_table["notable_type"]].append(i)
notable_types = list(notable_type_to_indices.keys())

maximum_num_ref = max(
    [len(per_input_indices) for per_input_indices in dedup_map.values()]
)

metrics_dict = dict()
bleu_scorer = BLEU()
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
rouge_scorer = _rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

all_predictions = []
if template_args.eval_specific_type > -1:
    type_list = [template_args.eval_specific_type]
else:
    type_list = list(range(num_types))
for type_i in type_list:
    type_indices = notable_type_to_indices[notable_types[type_i]]
    bleu_multi_references = [[] for _ in range(maximum_num_ref)]
    bert_multi_references = []
    # sample_i is index of the deduped dataset
    for sample_i in type_indices:
        sample_multi_references = []
        val_sample_input = eval_dataset.raw_datasets[template_args.evaluation_split][dedup_indices[sample_i]][
            "input_text"
        ]["context"]
        # val_sample_j is index of dataset before deduplication
        for val_sample_j in dedup_map[val_sample_input]:
            sample_multi_references.append(
                untokenize(
                    eval_dataset.raw_datasets[template_args.evaluation_split][val_sample_j][
                        "target_text"
                    ].lower()
                )
            )
        for ref_i in range(maximum_num_ref):
            bleu_multi_references[ref_i].append(
                sample_multi_references[ref_i]
                if ref_i < len(sample_multi_references)
                else None
            )
        bert_multi_references.append(sample_multi_references)

    if template_args.evaluation_split == "val":
        predictions = torch.load(
            f"{template_dir}/dev_output/type_{type_i}/all_return_ids.pt"
        )
    else:
        predictions = torch.load(
            f"{template_dir}/test_output/type_{type_i}/all_return_ids.pt"
        )
    predictions = [
        tokenizer.decode(p, skip_special_tokens=True).strip() for p in predictions
    ]

    bleu_score = bleu_scorer.corpus_score(predictions, bleu_multi_references).score
    bert_score = list(bert_scorer.score(predictions, bert_multi_references))
    bert_score = [t.mean().item() for t in bert_score]

    all_rouge_scores = [[], [], []]
    for pred, refs in zip(predictions, bert_multi_references):
        best_rouge = None
        for r in refs:
            temp_rouge = rouge_scorer.score(pred, r)
            if best_rouge is None or best_rouge['rougeL'].fmeasure < temp_rouge['rougeL'].fmeasure:
                best_rouge = temp_rouge
        all_rouge_scores[0].append(best_rouge["rouge1"].fmeasure)
        all_rouge_scores[1].append(best_rouge["rouge2"].fmeasure)
        all_rouge_scores[2].append(best_rouge["rougeL"].fmeasure)

    metrics_dict[type_i] = {
        "bleu": bleu_score,
        "bert_p": bert_score[0],
        "bert_r": bert_score[1],
        "bert_f": bert_score[2],
        "rouge1": np.mean(all_rouge_scores[0]),
        "rouge2": np.mean(all_rouge_scores[1]),
        "rougeL": np.mean(all_rouge_scores[2])
    }
    print(f"Type {type_i}: {metrics_dict[type_i]}")
    all_predictions += predictions

if len(type_list) == 8:
    bleu_multi_references = [[] for _ in range(maximum_num_ref)]
    bert_multi_references = []
    # val_sample_i is index of dataset before deduplication
    for val_sample_i in dedup_indices:
        sample_multi_references = []
        val_sample_input = eval_dataset.raw_datasets[template_args.evaluation_split][val_sample_i]["input_text"][
            "context"
        ]
        for val_sample_j in dedup_map[val_sample_input]:
            sample_multi_references.append(
                untokenize(
                    eval_dataset.raw_datasets[template_args.evaluation_split][val_sample_j]["target_text"].lower()
                )
            )
        bert_multi_references.append(sample_multi_references)
        for ref_i in range(maximum_num_ref):
            bleu_multi_references[ref_i].append(
                sample_multi_references[ref_i]
                if ref_i < len(sample_multi_references)
                else None
            )
    bleu_score = bleu_scorer.corpus_score(all_predictions, bleu_multi_references).score
    print(f"Overall BLEU Score: {bleu_score}.")

    bert_score = list(bert_scorer.score(all_predictions, bert_multi_references))
    bert_score = [t.mean().item() for t in bert_score]

    all_rouge_scores = [[], [], []]
    for pred, refs in zip(all_predictions, bert_multi_references):
        best_rouge = None
        for r in refs:
            temp_rouge = rouge_scorer.score(pred, r)
            if best_rouge is None or best_rouge['rougeL'].fmeasure < temp_rouge['rougeL'].fmeasure:
                best_rouge = temp_rouge
        all_rouge_scores[0].append(best_rouge["rouge1"].fmeasure)
        all_rouge_scores[1].append(best_rouge["rouge2"].fmeasure)
        all_rouge_scores[2].append(best_rouge["rougeL"].fmeasure)

    metrics_dict["full"] = {
        "bleu": bleu_score,
        "bert_p": bert_score[0],
        "bert_r": bert_score[1],
        "bert_f": bert_score[2],
        "rouge1": np.mean(all_rouge_scores[0]),
        "rouge2": np.mean(all_rouge_scores[1]),
        "rougeL": np.mean(all_rouge_scores[2])
    }

if template_args.evaluation_split == "val":
    torch.save(
        metrics_dict, template_dir / "dev_output" / "metrics.pt",
    )
else:
    torch.save(
        metrics_dict, template_dir / "test_output" / "metrics.pt",
    )