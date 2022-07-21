from dataclasses import dataclass
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.file_utils import PaddingStrategy
import random
import torch
import spacy
import benepar, spacy

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})


def create_infill_example(
    input_ids, labels, tokenizer, max_mask_tokens, min_mask_tokens, mask_start_idx=None
):
    # TODO: add argument
    max_mask_tokens = min(len(labels) - 2, max_mask_tokens)
    num_mask = random.randint(min_mask_tokens, max_mask_tokens)
    # exclud
    if mask_start_idx is None:
        mask_start_idx = random.randint(1, len(labels) - 1 - num_mask)
    mask_tokens = labels[mask_start_idx : mask_start_idx + num_mask]
    masked_context = (
        input_ids
        + labels[:mask_start_idx]
        + [tokenizer.mask_token_id]
        + labels[mask_start_idx + num_mask :]
    )
    masked_labels = mask_tokens + [tokenizer.eos_token_id]
    return masked_context, masked_labels


@dataclass
class DataCollatorForInfilling:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`
            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    max_mask_tokens: int = 10
    min_mask_tokens: int = 0

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        for feat in features:
            feat["input_ids"], feat["labels"] = create_infill_example(
                feat["input_ids"],
                feat["labels"],
                self.tokenizer,
                self.max_mask_tokens,
                self.min_mask_tokens,
            )
            del feat["attention_mask"]

        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                feature["labels"] = (
                    feature["labels"] + remainder
                    if padding_side == "right"
                    else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(
            self.model, "prepare_decoder_input_ids_from_labels"
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features


def enumerate_const(span):
    """
    This funciton returns all the child constituents in a document
    """
    ret = []
    if len(span._.labels) != 0 and span._.labels[0] != "S":
        ret.append((span._.labels, span, span.start, span.end))
    if len(list(span._.children)) != 0:
        for child in span._.children:
            child_ret = enumerate_const(child)
            if len(child_ret) != 0:
                ret.extend(child_ret)
    return ret


def index_const(fill_in_ids, const_ids):
    """
    Given the ids of the original sequence, return the index of the start of the constituency (in token idx space)
    """
    list_fill_in_ids = list(fill_in_ids.numpy())
    const_start_cands = [i for i, e in enumerate(list_fill_in_ids) if e == const_ids[0]]

    for start_cand in const_start_cands:
        cand_ngram = list_fill_in_ids[start_cand : start_cand + len(const_ids)]
        if (
            len(cand_ngram) == len(const_ids)
            and torch.tensor(cand_ngram).eq(torch.tensor(const_ids)).all()
        ):
            return start_cand
    raise ValueError("Not found: {}")


def link_const_chunks(chunk_token_spans, const_span):
    """
    a constituency is only valid if it does not cross chunk boundaries.
    if a constituency is valid, return the set of chunks it contains,
    else return empty
    """
    return_chunks = []
    chunk_i = 0
    while chunk_token_spans[chunk_i][0] < const_span[0]:
        chunk_i += 1
    if chunk_token_spans[chunk_i][0] != const_span[0]:
        return []
    while chunk_token_spans[chunk_i][1] <= const_span[1]:
        return_chunks.append((chunk_i, chunk_token_spans[chunk_i]))
        chunk_i += 1
    if len(return_chunks) > 0 and return_chunks[-1][1][1] != const_span[1]:
        return []
    else:
        return return_chunks


def compute_lexicalized_mask(alignment, chunk_indices, template_fill_in_ids, tokenizer):
    """
    Create a mask where the True locations represents chunks that are lexicalized.

    :param alignment:
    :param chunk_indices:
    :param template_fill_in_ids:
    :param tokenizer:
    :return:
    """
    lexicalized_mask = []
    for sample_i, sample_alignment in enumerate(alignment):
        sample_mask = []
        for chunk_align_i in chunk_indices[:-1]:
            token_i = sample_alignment[chunk_align_i]
            sample_mask.append(
                template_fill_in_ids[sample_i][token_i].item()
                not in tokenizer.field_start_set
            )
        lexicalized_mask.append(sample_mask)
    lexicalized_mask = torch.tensor(lexicalized_mask)
    return lexicalized_mask


def sort_valid_consts(
    tokenizer,
    template_ids,
    og_ids,
    og_alignment,
    template_chunk_indices,
    all_alignments,
    all_fill_in_ids,
    all_chunked_probs,
    lexicalized_only=True,
):
    """
    Given a template, sort all of its constituents by the average score of each constituent

    :param tokenizer:
    :param template_ids:
    :param og_ids:
    :param og_alignment:
    :param template_chunk_indices:
    :param all_alignments:
    :param all_fill_in_ids:
    :param all_chunked_probs:
    :param lexicalized_only:
    :return:
    """
    og_text = tokenizer.decode(og_ids, skip_special_tokens=True)
    doc = nlp(og_text)

    all_const = []
    for sent in doc.sents:
        all_const.extend(enumerate_const(sent))

    chunk_token_spans = []
    for chunk_start, chunk_end in zip(
        template_chunk_indices[:-2], template_chunk_indices[1:-1]
    ):
        # chunk_span in the token idx space
        chunk_span_start_id = og_alignment[1:][chunk_start] - 1
        chunk_span_end_id = og_alignment[1:][chunk_end] - 1
        chunk_token_spans.append((chunk_span_start_id, chunk_span_end_id))

    valid_consts = []
    for const in all_const:
        const_ids = tokenizer.encode(
            str(doc[const[-2] : const[-1]]), add_special_tokens=False
        )
        try:
            const_start = index_const(og_ids, const_ids)
        except:
            print("Not matched due to tokenization: ", tokenizer.decode(const_ids))
            continue
        const_end = const_start + len(const_ids)
        const_span = (const_start, const_end)

        linked_chunk_spans = link_const_chunks(chunk_token_spans, const_span)

        if len(linked_chunk_spans) != 0:
            const_align_ids = [
                template_chunk_indices[span[0]] for span in linked_chunk_spans
            ]
            const_chunk_ids = [span[0] for span in linked_chunk_spans]

            if lexicalized_only:
                lexicalized_mask = compute_lexicalized_mask(
                    all_alignments, template_chunk_indices, all_fill_in_ids, tokenizer
                )
                lexicalize_only_chunk_prob_mean = (
                    all_chunked_probs * lexicalized_mask.float()
                ).sum(dim=0)
                lexicalize_only_chunk_prob_mean = lexicalize_only_chunk_prob_mean / lexicalized_mask.float().sum(
                    dim=0
                )
                const_scores = (
                    lexicalize_only_chunk_prob_mean[const_chunk_ids].mean().item()
                )
            else:
                const_scores = (
                    all_chunked_probs[:, const_chunk_ids].mean(dim=0).mean().item()
                )
            valid_consts.append(
                (
                    const,
                    tokenizer.decode([template_ids[i] for i in const_align_ids],),
                    const_scores,
                    const_chunk_ids,
                )
            )
    return valid_consts
