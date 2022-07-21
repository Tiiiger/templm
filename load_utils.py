from transformers import AutoTokenizer, AutoConfig, BartForConditionalGeneration
from template_search_bart import TemplateSearchBART


def load_model_and_tokenizer(
    model_args,
    load_template_model=True,
    load_tokenizer=True,
    load_no_space_tokenizer=True,
):
    if load_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = None

    if load_no_space_tokenizer:
        no_space_tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=False,
        )
    else:
        no_space_tokenizer = None

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if load_template_model:
        model_class = TemplateSearchBART
    else:
        model_class = BartForConditionalGeneration
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    return model, tokenizer, no_space_tokenizer
