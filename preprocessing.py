import re


def untokenize(text):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.

    credit: https://stackoverflow.com/questions/21948019/python-untokenize-a-sentence
    """
    text = text.strip()
    text = text.replace("-lrb-", "(")
    text = text.replace("-rrb-", ")")
    text = text.replace("`` ", '" ').replace(" ''", ' "').replace(". . .", "...")
    text = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", text)
    text = re.sub(r" ([.,:;?!%]+)$", r"\1", text)
    return text.strip()


def e2e_preprocess_function(examples, data_args, tokenizer):
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    inputs = examples["meaning_representation"]
    targets = examples["human_reference"]
    inputs = [text.replace("[", " is ").replace("]", "") for text in inputs]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs,
        max_length=data_args.max_source_length,
        padding=padding,
        truncation=True,
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, padding=padding, truncation=True
        )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def synthbio_preprocess_function(examples, data_args, tokenizer):
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    raw_inputs = examples["input_text"]
    raw_targets = examples["target_text"]
    inputs = []
    for input_text in raw_inputs:
        input_data = ""
        for field_name, content in zip(
            input_text["table"]["column_header"], input_text["table"]["content"],
        ):
            if field_name in ["image", "website", "caption", "source"]:
                continue

            if input_data != "":
                input_data += ", "

            if len(content) > 0:
                if content[-1] in [",", "."]:
                    content = content[:-1]
                content = untokenize(content)
                input_data += f"{field_name} is {content}"
        inputs.append(input_data.lower())

    model_inputs = tokenizer(
        inputs,
        max_length=data_args.max_source_length,
        padding=padding,
        truncation=True,
    )

    targets = []
    for target_text in raw_targets:
        sen = target_text.split("\n")[0]
        targets.append(untokenize(sen).lower())

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, padding=padding, truncation=True
        )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
