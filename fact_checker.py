from datasets import load_dataset
import sys
import argparse
from pathlib import Path


def check_facts(output_dir, output_file_path, split_name):
    dataset_name = "e2e_nlg"
    raw_datasets = load_dataset(dataset_name)

    baseline_file = output_file_path

    print(baseline_file)
    check_human_ref = False

    seen_input_data = set()
    dedup_indices = []
    for i, dt in enumerate(raw_datasets[split_name]):
        if dt["meaning_representation"] not in seen_input_data:
            dedup_indices.append(i)
            seen_input_data.add(dt["meaning_representation"])

    raw_datasets[split_name] = raw_datasets[split_name].select(dedup_indices)

    raw_datasets[split_name] = raw_datasets[split_name].sort("meaning_representation")

    print(len(seen_input_data))
    print(raw_datasets[split_name])

    e2e_field_transformations = {
        ("food", "Fast food"): ["Fast food", "fast food"],
        ("familyFriendly", "yes"): [
            "is family friendly",
            "is kid friendly",
            "is children friendly",
            "is family-friendly",
            "is child friendly",
            "is a family friendly",
            "is a kid friendly",
            "is a children friendly",
            "is a family-friendly",
            "is a child friendly",
            "for a family friendly",
            "for a kid friendly",
            "for a children friendly",
            "for a family-friendly",
            "for a child friendly",
        ],
        ("familyFriendly", "no"): [
            "not family friendly",
            "not kid friendly",
            "not children friendly",
            "not family-friendly",
            "not child friendly",
            "non family-friendly",
            "non-family-friendly",
            "non family friendly",
            "non-family friendly",
            "non children friendly",
            "non child friendly",
        ],
        ("customer rating", "1 out of 5"): [
            "1 out of 5",
            "low customer rating",
            "one star",
            "1 star",
        ],
        ("customer rating", "3 out of 5"): [
            "3 out of 5",
            "customer rating is average",
            "average customer rating",
            "three star",
            "moderate customer rating",
            "3 star",
        ],
        ("customer rating", "5 out of 5"): [
            "5 out of 5",
            "high customer rating",
            "five star",
            "5 star",
        ],
        ("customer rating", "high"): [
            "5 out of 5",
            "high customer rating",
            "five star",
            "5 star",
        ],
        ("customer rating", "average"): [
            "3 out of 5",
            "customer rating is average",
            "average customer rating",
            "three star",
            "3 star",
        ],
        ("customer rating", "low"): [
            "1 out of 5",
            "low customer rating",
            "one star",
            "1 star",
        ],
        ("priceRange", "less than £20"): [
            "less than £20",
            "cheap",
            "low price range",
            "low-priced",
            "low priced",
        ],
        ("priceRange", "£20-25"): [
            "£20-25",
            "moderate price range",
            "average price range",
            "moderately priced",
            "moderate prices",
            "average priced",
        ],
        ("priceRange", "more than £30"): [
            "more than £30",
            "high price range",
            "high priced",
            "expensive",
            "price range is high",
        ],
        ("priceRange", "low"): ["low price range", "low-priced"],
        ("priceRange", "cheap"): ["cheap", "low price range", "low priced"],
        ("priceRange", "moderate"): [
            "moderate price range",
            "moderately priced",
            "price range is moderate",
            "moderate prices",
            "average prices",
        ],
        ("priceRange", "high"): [
            "high price range",
            "high priced",
            "expensive",
            "price range is high",
        ],
    }

    from collections import defaultdict
    from tqdm.auto import tqdm, trange

    field_possible_values = defaultdict(set)

    with open(baseline_file) as f:
        for idx, line in enumerate(f):
            for pair in raw_datasets[split_name][idx]["meaning_representation"].split(
                ","
            ):
                field, value = pair.split("[")
                field_possible_values[field.strip()].add(value.replace("]", ""))

    # print(field_possible_values.keys())
    # for field, values in field_possible_values.items():
    #     print(f"{field}: {values}")

    ### we lexicalize the fields here: e.g. ("familyFriendly", "no") => "not family friendly", "not kid friendly", etc.
    for field, values in field_possible_values.items():
        new_set = values.copy()
        for value in values:
            if (field, value) in e2e_field_transformations:
                new_set.remove(value)
                new_set.update(e2e_field_transformations[(field, value)])
        field_possible_values[field] = new_set

    from collections import defaultdict
    import re

    precision_indices = []
    precision_error_term = []

    recall_indices = []
    recall_error_terms = []

    with open(baseline_file) as f:
        for idx, line in enumerate(f):
            ref_line = (
                raw_datasets[split_name][idx]["human_reference"]
                if check_human_ref
                else line
            ).lower()
            curr_pairs = [
                pair.split("[")
                for pair in raw_datasets[split_name][idx][
                    "meaning_representation"
                ].split(",")
            ]
            curr_pairs_dict = defaultdict(set)
            for pair in curr_pairs:
                curr_pairs_dict[pair[0].strip()].add(pair[1].replace("]", ""))

            #         ## check all keywords that might appear falsely
            for field, values in field_possible_values.items():
                for value in values:
                    value = value.lower()

                    ## if they appear, check if they appear legally, i.e. in the current MR
                    #                 ref_line_words = [re.sub(r'[^\w\s]', '', word) for word in ref_line.split()]
                    #                 if value in ref_line_words:
                    if value in ref_line:
                        found = False

                        ## check against the MRs
                        for ref_key, ref_values in curr_pairs_dict.items():

                            ## ref_value might be "yes", need to transform: e.g.
                            ## ("familyFriendly", "no") => "not family friendly", "not kid friendly", etc.
                            for ref_value in ref_values:
                                if (ref_key, ref_value) in e2e_field_transformations:
                                    ref_value = e2e_field_transformations[
                                        (ref_key, ref_value)
                                    ]
                                else:
                                    ref_value = [ref_value]
                                for ref_values_single in ref_value:
                                    ref_values_single = ref_values_single.lower()
                                    if value == ref_values_single:
                                        found = True
                                        break

                        if not found:
                            if (field, value) == ('food', 'indian') and curr_pairs_dict["near"] == {'Raja Indian Cuisine'}:
                                continue
                            print(idx)
                            print(ref_line)
                            print(curr_pairs)
                            print((field, value))
                            print("=" * 80)
                            precision_indices.append(idx)
                            precision_error_term.append((field, value))

            ## check all keywords that should appear does appear
            for ref_key, ref_values in curr_pairs_dict.items():
                found = False
                for ref_value in ref_values:
                    if (ref_key, ref_value) in e2e_field_transformations:
                        ref_value = e2e_field_transformations[(ref_key, ref_value)]
                    else:
                        ref_value = [ref_value]
                    for ref_values_single in ref_value:
                        if ref_values_single.lower() in ref_line:
                            found = True
                            break
                if not found:
                    recall_indices.append(idx)
                    recall_error_terms.append((ref_key, tuple(ref_values)))

    error_terms_recall = defaultdict(set)
    for elem in zip(recall_indices, recall_error_terms):
        error_terms_recall[elem[0]].add(elem[1])

    bad_recall = []
    bad_recall_indices = []

    with open(baseline_file) as f:
        for idx, data in enumerate(zip(raw_datasets[split_name], f)):
            if idx in error_terms_recall:
                bad_recall_indices.append(idx)
                bad_recall.append(data)

    print("# Recall Error: ", len(bad_recall))
    # for idx, incs in enumerate(bad_recall):
    #     print(f"Index: {bad_recall_indices[idx]} \n Train: {incs[0]}, \n")
    #     if not check_human_ref:
    #         print(f"Baseline: {incs[1]}")
    #     print(f" Error Term: {error_terms_recall[bad_recall_indices[idx]]} \n")

    error_terms_precision = defaultdict(set)
    for elem in zip(precision_indices, precision_error_term):
        error_terms_precision[elem[0]].add(elem[1])

    inconsistencies = []
    idx_inconsistencies = []

    with open(baseline_file) as f:
        for idx, data in enumerate(zip(raw_datasets[split_name], f)):
            if idx in error_terms_precision:
                idx_inconsistencies.append(idx)
                inconsistencies.append(data)

    print("# Precision Error: ", len(inconsistencies))
    precision_error_fields = []
    for precision_error in error_terms_precision.values():
        for p_e in precision_error:
            precision_error_fields.append(p_e[0])
    from collections import Counter

    # print(Counter(precision_error_fields).most_common())
    # for idx, incs in enumerate(inconsistencies):
    #     print(f"Index: {idx_inconsistencies[idx]} \n Train: {incs[0]},")
    #     if not check_human_ref:
    #         print(f"\n Baseline: {incs[1]}")
        # print(f" Error Term: {error_terms_precision[idx_inconsistencies[idx]]} \n")

    with open(Path(output_dir) / "fact_checker.txt", "w") as f:
        f.write(f"precision_e: {len(inconsistencies)}\n")
        f.write(f"recall_e: {len(bad_recall)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_file_path", type=str)
    parser.add_argument("--split_name", type=str)
    args = parser.parse_args()
    check_facts(args.output_dir, args.output_file_path, args.split_name)
