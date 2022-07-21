from datasets import load_dataset
from collections import defaultdict
import torch
from pathlib import Path
from preprocessing import (
    e2e_preprocess_function,
    synthbio_preprocess_function,
    untokenize,
)
from functools import partial
from tqdm.auto import tqdm, trange
import numpy as np
import random
from collections import Counter
import os
from field_transformation import (
    e2e_field_transformation,
    sb_field_transformation,
)


# noinspection PyAttributeOutsideInit
class TabularData:
    def __init__(
        self,
        dataset_name,
        split_name,
        tokenizer,
        no_space_tokenizer,
        data_args,
        training_args,
        field_tokens_cutoff=5,
    ):
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.tokenizer = tokenizer
        self.no_space_tokenizer = no_space_tokenizer

        self.data_args = data_args
        self.training_args = training_args

        raw_datasets = load_dataset(dataset_name,)

        # deduplicate based on meaning representation for E2E
        if "e2e" in dataset_name:
            self.mode = "e2e"
            seen_input_data = set()
            if data_args.dedup_input:
                dedup_indices = []
                for i, dt in enumerate(raw_datasets[self.split_name]):
                    if dt["meaning_representation"] not in seen_input_data:
                        dedup_indices.append(i)
                        seen_input_data.add(dt["meaning_representation"])
                raw_datasets[self.split_name] = raw_datasets[self.split_name].select(
                    dedup_indices
                )
                os.makedirs(training_args.output_dir, exist_ok=True)
                torch.save(
                    dedup_indices,
                    Path(training_args.output_dir, f"{split_name}_dedup_indices.pt"),
                )

            # we sort the development set
            if split_name in ["validation", "test"]:
                raw_datasets[self.split_name] = raw_datasets[self.split_name].sort(
                    "meaning_representation"
                )
        elif "synthbio" in dataset_name:
            self.mode = "wiki_bio"
            seen_input_data = set()
            dedup_map = dict()
            if data_args.dedup_input:
                dedup_indices = []
                for i, dt in enumerate(raw_datasets[self.split_name]):
                    if dt["input_text"]["context"] not in seen_input_data:
                        dedup_indices.append(i)
                        dedup_map[dt["input_text"]["context"]] = [i]
                        seen_input_data.add(dt["input_text"]["context"])
                    else:
                        dedup_map[dt["input_text"]["context"]].append(i)
                raw_datasets[self.split_name] = raw_datasets[self.split_name].select(
                    dedup_indices
                )
                self.dedup_map = dedup_map
                self.dedup_indices = dedup_indices

            self.field_keep_set = set()

            for field, field_count in field_counter.items():
                if field_count > 0.01 * len(raw_datasets["train"]):
                    self.field_keep_set.add(field)

        self.raw_datasets = raw_datasets

        self.data_args = data_args
        self.training_args = training_args

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = raw_datasets[self.split_name].column_names

        if "e2e" in dataset_name:
            preprocess_function = partial(
                e2e_preprocess_function, data_args=data_args, tokenizer=tokenizer
            )
        elif "synthbio" in dataset_name:
            preprocess_function = partial(
                synthbio_preprocess_function, data_args=data_args, tokenizer=tokenizer
            )
        else:
            raise ValueError("Invalid dataset name.")

        self.preprocess_function = preprocess_function

        processed_dataset = raw_datasets[self.split_name]
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            self.processed_dataset = processed_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

        self.get_type_info(field_tokens_cutoff=field_tokens_cutoff)
        self.tokenizer.save_pretrained(training_args.output_dir)

    def few_shot_sample(self, data_per_type):
        np.random.seed(self.training_args.seed)
        few_shot_indices = []

        _combintation_to_indices = list(self.combination_to_indices.items())
        self.indices_to_combination = dict()
        for i in range(len(_combintation_to_indices)):
            prev_len = len(few_shot_indices)
            combination, indices = _combintation_to_indices[i]
            if len(indices) >= data_per_type:
                combination_indices = np.random.choice(
                    list(indices), data_per_type, replace=False
                ).tolist()
            else:
                combination_indices = indices
            few_shot_indices.extend(combination_indices)
            combination_new_indices = list(
                range(prev_len, prev_len + len(combination_indices))
            )
            self.combination_to_indices[combination] = combination_new_indices
            for i in combination_new_indices:
                self.indices_to_combination[i] = combination

        torch.save(
            few_shot_indices,
            Path(
                self.training_args.output_dir, f"{self.split_name}_few_shot_indices.pt"
            ),
        )
        self.processed_dataset = self.processed_dataset.select(few_shot_indices)
        self.all_transformed_data = [
            self.all_transformed_data[i] for i in few_shot_indices
        ]
        self.field_possible_values = defaultdict(set)
        self.field_to_values = [self.field_to_values[i] for i in few_shot_indices]
        for data_table in self.field_to_values:
            for field_name, field_content in data_table.items():
                self.field_possible_values[field_name].add(field_content)
        self.field_dicts = [self.field_dicts[i] for i in few_shot_indices]

    def get_type_info(self, field_tokens_cutoff=5):
        self.all_field_names = set()
        if "e2e" in self.dataset_name:
            for dt in self.raw_datasets[self.split_name]:
                data_table = dt["meaning_representation"]
                data_table = data_table.split(", ")
                data_table = [t.split("[")[0] for t in data_table]
                self.all_field_names.update(data_table)
        else:
            for dt in self.raw_datasets[self.split_name]:
                data_table = dt["input_text"]["table"]["column_header"]
                self.all_field_names.update(data_table)

        field_to_values = list()
        field_to_tokens = list()
        combination_to_indices = defaultdict(set)  # type_combination to indices
        indices_to_combination = dict()  # indices to type_combination
        field_possible_values = defaultdict(
            set
        )  # field names mapped to potential values
        temp_field_max_lens = defaultdict(
            int
        )  # maximum lens of a field, before truncating
        all_tranformed_data = list()  # data table after field transformation
        for i, sample in tqdm(
            enumerate(self.raw_datasets[self.split_name]),
            total=len(self.raw_datasets[self.split_name]),
            desc="tokenizing field tokens",
        ):
            field_values = dict()
            field_tokens = dict()
            if "e2e" in self.dataset_name:
                data_table = sample["meaning_representation"]
                data_table = data_table.split(", ")
                data_table = dict([tuple(t[:-1].split("[")) for t in data_table])
                for name, content in data_table.items():
                    # record original content
                    field_values[name] = content
                    field_possible_values[name].add(content)

                data_table = e2e_field_transformation(data_table, self.data_args.complex_field_transformation)
                all_tranformed_data.append(data_table)
            elif "synthbio" in self.dataset_name:
                input_table = dict(
                    zip(
                        sample["input_text"]["table"]["column_header"],
                        sample["input_text"]["table"]["content"],
                    )
                )
                data_table = sb_field_transformation(input_table)
                all_tranformed_data.append(data_table)
                for name, content in data_table.items():
                    if len(content) > 0:
                        random_content = random.choice(list(content))
                        field_values[name] = random_content
                        field_possible_values[name].add(random_content)
            else:
                raise ValueError("invalid dataset name")

            for name, content in data_table.items():
                content_tokens = set()
                for c in content:
                    if (
                        "wiki_bio" in self.dataset_name
                        or "synthbio" in self.dataset_name
                    ):
                        c = c.lower()
                    toks = tuple(self.tokenizer.encode(c, add_special_tokens=False))
                    content_tokens.add(toks)
                    if (
                        "wiki_bio" in self.dataset_name
                        or "synthbio" in self.dataset_name
                    ):
                        no_space_toks = tuple(
                            self.no_space_tokenizer.encode(c, add_special_tokens=False)
                        )
                        content_tokens.add(no_space_toks)
                if len(content_tokens) > 0:
                    field_tokens[name] = content_tokens
                    temp_field_max_lens[name] = max(
                        temp_field_max_lens[name],
                        max([len(c) for c in content_tokens]),
                    )

            type_combination = tuple(sorted(field_values.keys()))
            combination_to_indices[type_combination].add(i)
            indices_to_combination[i] = type_combination
            field_to_values.append(field_values)
            field_to_tokens.append(field_tokens)

        self.combination_to_indices = {
            k: sorted(v) for k, v in combination_to_indices.items()
        }
        self.indices_to_combination = indices_to_combination
        self.field_possible_values = {
            k: sorted(v) for k, v in field_possible_values.items()
        }
        self.field_to_values = field_to_values
        self.field_to_tokens = field_to_tokens
        self.field_max_lens = dict()
        self.all_transformed_data = all_tranformed_data

        # only modify the tokenizer if we are in training mode
        if self.split_name == "train":
            multi_special_tokens = []
            for field_name, field_max_len in temp_field_max_lens.items():
                field_max_len = min(field_max_len, field_tokens_cutoff)
                for i in range(field_max_len):
                    multi_special_tokens.append(f"<{field_name}-{i}>")
                self.field_max_lens[field_name] = field_max_len

            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": multi_special_tokens}
            )

            next_token_map = dict()
            field_start_set = set()
            field_cutoff_set = set()
            for field_name, field_max_len in temp_field_max_lens.items():
                field_tokens = self.tokenizer.convert_tokens_to_ids(
                    [
                        f"<{field_name}-{i}>"
                        for i in range(min(field_max_len, field_tokens_cutoff))
                    ]
                )
                field_start_set.add(field_tokens[0])
                if field_max_len > field_tokens_cutoff:
                    field_cutoff_set.add(field_tokens[field_tokens_cutoff - 1])
                next_token_map.update(
                    {
                        p_tok: n_tok
                        for (p_tok, n_tok) in zip(field_tokens[:-1], field_tokens[1:])
                    }
                )
            all_field_tokens = self.tokenizer.convert_tokens_to_ids(
                multi_special_tokens
            )
            # Tianyi: monkey patching for now. this is super ugly but I am lazy
            self.tokenizer.all_field_tokens = set(all_field_tokens)
            self.tokenizer.field_cutoff_set = field_cutoff_set
            self.tokenizer.field_start_set = field_start_set
            self.tokenizer.next_field_token_map = next_token_map

        self.field_dicts = []
        for field_dict in tqdm(field_to_tokens, desc="preparing field dicts"):
            linearized_field_dict = defaultdict(list)
            for field_name, tuple_token_ids in field_dict.items():
                max_lex_lens = max([len(token_ids) for token_ids in tuple_token_ids])
                all_run_over_tokens = []  # run over tokens for different rephrases
                for token_ids in tuple_token_ids:
                    # for different rephrases
                    run_over_tokens = []  # run over tokens for a single phrasing
                    for i in range(max_lex_lens):
                        if i > len(token_ids) - 1:
                            token_id = -1
                        else:
                            token_id = token_ids[i]
                        if i < field_tokens_cutoff:
                            field_name_token = self.tokenizer._convert_token_to_id_with_added_voc(
                                f"<{field_name}-{i}>"
                            )
                            if field_name_token == self.tokenizer.unk_token_id:
                                # field is longer than seen in the training set: can only appear in valiation set
                                assert self.split_name != "train"
                                continue
                            linearized_field_dict[field_name_token].append(token_id)
                        else:
                            run_over_tokens.append(token_id)
                    if len(run_over_tokens) != 0:
                        all_run_over_tokens.append(run_over_tokens)
                if len(all_run_over_tokens) != 0:
                    assert len(all_run_over_tokens) == len(tuple_token_ids)
                    cutoff_token = self.tokenizer._convert_token_to_id_with_added_voc(
                        f"<{field_name}-{field_tokens_cutoff-1}>"
                    )
                    linearized_field_dict[
                        (cutoff_token, "runover")
                    ] = all_run_over_tokens
            linearized_field_dict = dict(linearized_field_dict)
            self.field_dicts.append(linearized_field_dict)

    def sample_field_combination(self, field_combination, num_inputs):
        assert self.mode == "e2e"
        sample_input_ids = []
        sample_field_dicts = []
        sample_raw_field_dicts = []
        possible_combination = np.prod(
            [
                len(self.field_possible_values[field_name])
                for field_name in field_combination
            ]
        )
        value_combinations = set()
        if possible_combination >= num_inputs:
            while len(sample_input_ids) < num_inputs:
                value_combination = tuple(
                    np.random.choice(list(self.field_possible_values[field_name]))
                    for field_name in field_combination
                )
                if value_combination not in value_combinations:
                    value_combinations.add(value_combination)
                    sample_input_id, sample_field_dict = self._create_new_input(
                        field_combination, value_combination
                    )
                    sample_input_ids.append(sample_input_id)
                    sample_field_dicts.append(sample_field_dict)
                    sample_raw_field_dicts.append(
                        e2e_field_transformation(
                            dict(zip(field_combination, value_combination)),
                            self.data_args.complex_field_transformation
                        )
                    )
        else:
            value_combinations = []
            for field_name in field_combination:
                new_value_combinations = []
                for value in self.field_possible_values[field_name]:
                    if len(value_combinations) == 0:
                        new_value_combinations.append([value])
                    else:
                        for value_combination in value_combinations:
                            new_value_combinations.append(value_combination + [value])
                value_combinations = new_value_combinations

            sample_input_ids = []
            sample_field_dicts = []
            sample_raw_field_dicts = []
            for value_combination in value_combinations:
                sample_input_id, sample_field_dict = self._create_new_input(
                    field_combination, value_combination
                )
                sample_input_ids.append(sample_input_id)
                sample_field_dicts.append(sample_field_dict)
                sample_raw_field_dicts.append(
                    e2e_field_transformation(
                        dict(zip(field_combination, value_combination)),
                        self.data_args.complex_field_transformation
                    )
                )

        return sample_input_ids, sample_field_dicts, sample_raw_field_dicts

    def _create_new_input(self, field_combination, value_combination):
        input_sen = ""
        sample_field_dict = defaultdict(list)
        data_table = dict(zip(field_combination, value_combination))
        data_table = e2e_field_transformation(data_table, self.data_args.complex_field_transformation)
        for field_name, value in zip(field_combination, value_combination):
            if input_sen != "":
                input_sen += ", "
            input_sen += f"{field_name} is {value}"

        for field_name, field_content in data_table.items():
            list_content_ids = [
                self.tokenizer.encode(c, add_special_tokens=False)
                for c in field_content
            ]
            max_lex_lens = max([len(content_ids) for content_ids in list_content_ids])
            for content_ids in list_content_ids:
                for i in range(max_lex_lens):
                    field_token = self.tokenizer.encode(
                        f"<{field_name}-{i}>", add_special_tokens=False
                    )
                    assert len(field_token) == 1
                    field_token = field_token[0]
                    if i > len(content_ids) - 1:
                        content_token = -1
                    else:
                        content_token = content_ids[i]
                    sample_field_dict[field_token].append(content_token)
        sample_input_ids = self.tokenizer.encode(input_sen)
        return sample_input_ids, sample_field_dict
