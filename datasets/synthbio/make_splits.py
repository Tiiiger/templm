import json
from collections import defaultdict, Counter
import torch

with open("./SynthBio.json", "r") as f:
    raw_data = json.load(f)

i2t = {i: raw_data[i]["notable_type"] for i in range(len(raw_data))}
t2i = defaultdict(list)
for i, sample in enumerate(raw_data):
    t2i[sample["notable_type"]].append(i)

train_ratio, val_ratio, test_ratio = 1500 / 2237, 367 / 2237, 368 / 2237

tr_data = []
val_data = []
te_data = []
for notable_type, indices in t2i.items():
    type_count = len(indices)
    tr_count, val_count, te_count = (
        int(train_ratio * type_count),
        int(val_ratio * type_count),
        int(test_ratio * type_count),
    )
    while (tr_count + val_count + te_count) < type_count:
        tr_count += 1
    while (tr_count + val_count + te_count) > type_count:
        tr_count -= 1
    _indices = torch.randperm(len(indices))
    _tr_indices = _indices[:tr_count]
    _val_indices = _indices[tr_count : tr_count + val_count]
    _te_indices = _indices[-te_count:]
    tr_indices = [indices[i] for i in _tr_indices]
    val_indices = [indices[i] for i in _val_indices]
    te_indices = [indices[i] for i in _te_indices]
    for i in tr_indices:
        tr_data.append(raw_data[i])
    for i in val_indices:
        val_data.append(raw_data[i])
    for i in te_indices:
        te_data.append(raw_data[i])

with open("./train.json", "w") as f:
    json.dump(tr_data, f)
with open("./val.json", "w") as f:
    json.dump(val_data, f)
with open("./test.json", "w") as f:
    json.dump(te_data, f)
