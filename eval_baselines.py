import os
import torch
from pathlib import Path

from transformers import (
    BartForConditionalGeneration,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)
from model_args import ModelArguments
from data_args import DataTrainingArguments
from template_args import TemplateArguments
import utils
from data import TabularData
from tqdm import trange
from load_utils import *
import datasets

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

model, tokenizer, no_space_tokenizer = load_model_and_tokenizer(
    model_args, load_template_model=False
)
model.cuda()

os.makedirs(training_args.output_dir, exist_ok=True)
if template_args.evaluation_split == "val":
    if "e2e" in data_args.dataset_name:
        val_split_name = "validation"
    elif "synthbio" in data_args.dataset_name:
        val_split_name = "val"
    else:
        raise ValueError("Invalid dataset: ", data_args.dataset_name)
elif template_args.evaluation_split == "test":
    val_split_name = "test"
else:
    raise ValueError("Invalid evaluation split: ", template_args.evaluation_split)


# inputs are deduplicated here
assert data_args.dedup_input == False
eval_dataset = TabularData(
    data_args.dataset_name,
    val_split_name,
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
for i, dt in enumerate(eval_dataset.raw_datasets[val_split_name]):
    if "synthbio" in data_args.dataset_name:
        dedup_field = dt["input_text"]["context"]
    else:
        dedup_field = dt["meaning_representation"]
    if dedup_field not in seen_input_data:
        dedup_indices.append(i)
        dedup_map[dedup_field] = [i]
        seen_input_data.add(dedup_field)
    else:
        dedup_map[dedup_field].append(i)

eval_input_ids = eval_dataset.processed_dataset[dedup_indices]["input_ids"]
eval_input_ids = utils.pad_list(eval_input_ids, tokenizer.pad_token_id)
eval_input_ids = torch.LongTensor(eval_input_ids)

# Get baseline BART results
baseline_results = []
for batch_start in trange(
    0, eval_input_ids.size(0), template_args.inference_batch_size
):
    batch_input_ids = eval_input_ids[
        batch_start : batch_start + template_args.inference_batch_size
    ].cuda()
    output_ids = model.generate(
        batch_input_ids,
        max_length=template_args.max_decode_steps,
        num_beams=data_args.num_beams,
        num_return_sequences=1,
        length_penalty=template_args.length_penalty,
        early_stopping=False,
    )
    for sen_ids in output_ids:
        baseline_results.append(
            tokenizer.decode(sen_ids, skip_special_tokens=True).strip()
        )

if template_args.evaluation_split == "val":
    output_dir = Path(training_args.output_dir) / "dev_output"
else:
    output_dir = Path(training_args.output_dir) / "test_output"

output_dir.mkdir(exist_ok=True)
with open(output_dir / "baseline_output.txt", "w") as f:
    for t in baseline_results:
        f.write(t + "\n")

if "synthbio" in data_args.dataset_name:
    import numpy as np
    from sacrebleu import BLEU
    from bert_score import BERTScorer
    from rouge_score import rouge_scorer as _rouge_scorer
    from preprocessing import untokenize
    from collections import defaultdict

    bleu_scorer = BLEU()
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    rouge_scorer = _rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    metrics_dict = dict()

    maximum_num_ref = max(
        [len(per_input_indices) for per_input_indices in dedup_map.values()]
    )
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
    bleu_score = bleu_scorer.corpus_score(baseline_results, bleu_multi_references).score
    bert_score = list(bert_scorer.score(baseline_results, bert_multi_references))
    bert_score = [t.mean().item() for t in bert_score]

    all_rouge_scores = [[], [], []]
    for pred, refs in zip(baseline_results, bert_multi_references):
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
    print(metrics_dict)

    deduped_dataset = eval_dataset.raw_datasets["val"].select(dedup_indices)
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

    for type_i in range(8):
        type_indices = notable_type_to_indices[notable_types[type_i]]

        predictions = [baseline_results[i] for i in type_indices]

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

        torch.save(
            metrics_dict, output_dir / "metrics.pt",
        )