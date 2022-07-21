import os
from pathlib import Path
import torch
from transformers import DataCollatorForSeq2Seq
import re
import numpy as np


def get_datacollator(tokenizer, model, data_args, training_args):
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    return data_collator


def pad_list(sequences, pad_token_id):
    max_len = max([len(x) for x in sequences])
    for i, input_ids in enumerate(sequences):
        sequences[i] = input_ids + [pad_token_id] * (max_len - len(input_ids))
    sequences = torch.LongTensor(sequences)
    return sequences


def load_templates(template_dir, template_args, mode="e2e"):
    if not isinstance(template_dir, Path):
        template_dir = Path(template_dir)
    if not template_args.eval_typed_templates:
        raise NotImplemented
    else:
        num_types = len([d for d in os.listdir(template_dir) if d.startswith("type")])
        all_templates = dict()
        type_combination_indices = dict()

        if template_args.eval_specific_type > -1:
            type_list = [template_args.eval_specific_type]
        else:
            type_list = list(range(num_types))
        for type_i in type_list:
            if mode == "e2e":
                dict_key = torch.load(
                    template_dir / f"type_{type_i}" / "field_combination.pt"
                )
            else:
                dict_key = type_i
            type_combination_indices[dict_key] = type_i
            if template_args.local_search_prune_topk is None:
                all_templates[dict_key] = torch.load(
                    template_dir / f"type_{type_i}" / "templates.pt"
                )
            else:
                templates = torch.load(template_dir / f"type_{type_i}" / "templates.pt")
                token_scores = torch.load(
                    template_dir / f"type_{type_i}" / "token_scores.pt"
                )
                score_stats = [
                    (
                        s.sum(dim=-1)[0]
                        / s.lt(0).float().sum(dim=-1).pow(template_args.length_penalty)
                    )
                    .mean()
                    .item()
                    for s in token_scores
                ]
                rank_ids = np.argsort(score_stats)
                all_templates[dict_key] = []
                for i in range(len(rank_ids)):
                    try:
                        template = templates[rank_ids[-(i + 1)]]
                        if len(template) > 3:
                            all_templates[dict_key].append(template)
                        if (
                            len(all_templates[dict_key])
                            == template_args.local_search_prune_topk
                        ):
                            break
                    except IndexError:
                        continue

    return all_templates, type_combination_indices


def exact_match_template(
    target_text,
    raw_field_dict,
    field_max_lens,
    tokenizer,
    no_space_tokenizer,
    mode="e2e",
):
    """
    Implements Delexicalization for generating templates.
    """
    tokenized_raw_field_dict = {
        k: [
            tokenizer.encode(
                tok if mode == "e2e" else tok.lower(), add_special_tokens=False
            )
            for tok in tok_list
        ]
        for k, tok_list in raw_field_dict.items()
    }
    if mode == "synthbio":
        for k, tok_list in raw_field_dict.items():
            for tok in tok_list:
                tokenized_raw_field_dict[k].append(
                    no_space_tokenizer.encode(tok.lower(), add_special_tokens=False)
                )

    to_replace = []
    max_field_len = max(
        [
            max([len(c) for c in content_list])
            for content_list in tokenized_raw_field_dict.values()
        ]
    )
    cand_ngrams = {}
    for n in range(1, max_field_len + 3):
        cand_ngrams[n] = []
        for i in range(0, len(target_text) + 1 - n):
            cand_ngrams[n].append((i, target_text[i : i + n]))

    for field_name, field_contents in tokenized_raw_field_dict.items():
        for field_content in field_contents:
            field_cands = cand_ngrams[len(field_content)]
            for cand_i, cand_ngram in field_cands:
                if tuple(cand_ngram) == tuple(field_content):
                    to_replace.append(
                        (
                            cand_i,
                            len(cand_ngram),
                            field_name,
                            tokenizer.decode(cand_ngram),
                        )
                    )
    to_replace = sorted(to_replace, key=lambda x: x[1], reverse=True)
    for i in range(len(to_replace)):
        if to_replace[i] is None:
            continue
        replace_start, replace_len, field_name, _ = to_replace[i]
        field_max_len = field_max_lens[field_name]
        field_non_terminals = tokenizer.encode(
            "".join([f"<{field_name}-{i}>" for i in range(field_max_len)]),
            add_special_tokens=False,
        )
        target_text = (
            target_text[:replace_start]
            + field_non_terminals
            + target_text[replace_start + replace_len :]
        )
        for j in range(i + 1, len(to_replace)):
            if to_replace[j] is None:
                continue
            (
                later_replace_start,
                later_replace_len,
                later_field_name,
                later_ngram,
            ) = to_replace[j]
            if later_replace_start < replace_start:
                if later_replace_start + later_replace_len > replace_start:
                    to_replace[j] = None
            elif (
                later_replace_start >= replace_start
                and later_replace_start < replace_start + replace_len
            ):
                to_replace[j] = None
            else:
                to_replace[j] = (
                    later_replace_start - replace_len + field_max_len,
                    later_replace_len,
                    later_field_name,
                    later_ngram,
                )

    return [2, 0] + target_text + [2]


def decode_and_clean_template(template, tokenizer, field_to_maxlen_dict):
    clean_template = tokenizer.decode(template)
    clean_template = re.sub("<(s|/s|pad)>", "", clean_template)
    tag_set = {
        re.sub(r"[<>\-\d+]", "", w)
        for w in re.findall(r"<[a-zA-Z_\s]+-\d+>", clean_template)
    }  # detect all the tags that look like <{some characters}_{digit}>
    for tag in tag_set:
        maxlen_tag = field_to_maxlen_dict[tag]
        clean_template = re.sub(
            re.compile("(?!>)" + f"<{tag}-\d>" * maxlen_tag),
            f" [{tag.upper()}]",
            clean_template,
        )
    clean_template = clean_template.replace("  ", " ").strip()
    return clean_template


def compute_logprobs(model, input_ids, output_ids, pad_token_id, batch_size=16):
    output_logprobs = []
    for batch_start in range(0, output_ids.size(0), batch_size):
        batch_input_ids = input_ids[batch_start : batch_start + batch_size].to(
            model.device
        )
        batch_output_ids = output_ids[batch_start : batch_start + batch_size].to(
            model.device
        )
        batch_outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_input_ids.ne(pad_token_id),
            decoder_input_ids=batch_output_ids,
            return_dict=True,
        )

        batch_output_logprobs = (
            batch_outputs.logits[:, :-1]
            .log_softmax(dim=-1)
            .gather(index=batch_output_ids[:, 1:].unsqueeze(-1).cuda(), dim=-1)
            .squeeze(-1)
        )
        batch_output_logprobs = batch_output_logprobs * (
            batch_output_ids[:, 1:].cuda().ne(pad_token_id)
        )

        output_logprobs.append(batch_output_logprobs)
        del batch_outputs
    output_logprobs = torch.cat(output_logprobs)
    return output_logprobs


def get_field_name(field_id, tokenizer):
    return re.sub(r"[<>\-\d+]", "", tokenizer._convert_id_to_token(field_id))


def align_and_chunk_logprobs(template, tokenizer, template_alignments, output_logprobs):
    if not isinstance(template, list):
        template = template.cpu().tolist()
    num_non_pad = torch.tensor(template).ne(tokenizer.pad_token_id).sum().item()

    all_aligned_logprobs = []
    for i in range(output_logprobs.size(0)):
        aligned_logprobs = []
        for j in range(num_non_pad):
            s, e = template_alignments[i][j : j + 2]
            if s == 0 and e == 1:
                aligned_logprobs.append(-1)
            else:
                aligned_logprobs.append(output_logprobs[i, s - 1 : e - 1].sum().item())
        all_aligned_logprobs.append(aligned_logprobs)
    all_aligned_logprobs = torch.tensor(all_aligned_logprobs)

    template_chunk_indices = []
    template_curr = 0
    while template_curr < num_non_pad:
        template_chunk_indices.append(template_curr)
        token_id = template[template_curr]
        if token_id not in tokenizer.all_field_tokens:
            template_curr += 1
        else:
            field = get_field_name(token_id, tokenizer)
            while (
                token_id in tokenizer.all_field_tokens
                and get_field_name(token_id, tokenizer) == field
            ):
                template_curr += 1
                token_id = template[template_curr]
    template_chunk_indices.append(template_curr)

    all_chunked_logprobs = []
    for i, chunk_start in enumerate(template_chunk_indices[:-1]):
        chunk_end = template_chunk_indices[i + 1]
        all_chunked_logprobs.append(
            all_aligned_logprobs[:, chunk_start:chunk_end].sum(dim=-1)
        )
    all_chunked_logprobs = torch.stack(all_chunked_logprobs, dim=-1)

    return all_aligned_logprobs, all_chunked_logprobs, template_chunk_indices
