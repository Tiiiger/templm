import os
import datasets
import json


class SynthBio(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                "input_text": {
                    "table": datasets.Sequence(
                        {
                            "column_header": datasets.Value("string"),
                            "content": datasets.Value("string"),
                        }
                    ),
                    "context": datasets.Value("string"),
                },
                "target_text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description="",
            features=features,
            supervised_keys=("serialized_attrs", "target_text"),
            homepage="",
            license="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_path = os.path.join(
            "/u/scr/tianyizhang/projects/tempretrain/datasets/synthbio"
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split("train"),
                gen_kwargs={"json_file": os.path.join(data_path, "train.json"),},
            ),
            datasets.SplitGenerator(
                name=datasets.Split("val"),
                gen_kwargs={"json_file": os.path.join(data_path, "val.json"),},
            ),
            datasets.SplitGenerator(
                name=datasets.Split("test"),
                gen_kwargs={"json_file": os.path.join(data_path, "test.json"),},
            ),
        ]

    def _generate_examples(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        count = -1
        for sample in data:
            for bio in sample["biographies"]:
                count += 1
                new_sample = dict()
                new_sample["target_text"] = bio
                attrs = []
                for field_name, field_val in sample["attrs"].items():
                    attrs.append({"column_header": field_name, "content": field_val})
                attrs.append(
                    {
                        "column_header": "notable_type",
                        "content": sample["notable_type"],
                    }
                )
                new_sample["input_text"] = {
                    "table": attrs,
                    "context": sample["serialized_attrs"],
                }
                new_sample["target_text"] = bio
                yield count, new_sample
