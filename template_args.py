from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TemplateArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    inference_batch_size: int = field(default=64,)
    max_decode_steps: int = field(default=100,)
    length_penalty: float = field(default=0)
    group_strategy: str = field(default="first")
    hallucinate_aug: bool = field(default=False)
    compute_baseline: bool = field(default=False)
    eval_typed_templates: bool = field(default=False)
    eval_postprocess: bool = field(default=False)
    search_crit: str = field(default="prob")
    local_search_baseline: bool = field(default=False)
    local_search_augment: int = field(default=0)
    local_search_augment_type: bool = field(default=False)
    local_search_compute_stats: bool = field(default=False)
    local_search_prune_topk: int = field(default=None)
    cluster_feature_choice: int = field(default=1)
    augment_size: int = field(default=500)
    refinement_lex_only: bool = field(default=True)
    refinement_topn_delex: int = field(default=5)
    refinement_const_cutoff: float = field(default=-2)
    refinement_template_dir: str = field(
        default="/u/scr/tianyizhang/output/tempretrain/synthbio_templates/exact_local_new/baseline"
    )
    refinement_infill_model_path: str = field(
        default="/u/scr/tianyizhang/output/tempretrain/bart_synthbio_infill_e20"
    )
    field_tokens_cutoff: int = field(default=5)
    eval_specific_type: int = field(default=-1)
    evaluation_split: str = field(default="val")
    random_selection_inference: bool = field(default=False)
