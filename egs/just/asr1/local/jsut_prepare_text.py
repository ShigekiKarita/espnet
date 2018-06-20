#!/usr/bin/env python3
import re
from glob import glob
import os

# ex: ..../downloads/jsut_ver1.1/basic5000/transcript_utf8.txt
path_list = glob("downloads/jsut_ver1.1/*/transcript_utf8.txt")
txt_dict = dict()
for path in path_list:
    d = path.split("/")[-2]
    txts = dict()
    with open(path, "r") as f:
        for txt in f:
            key = txt.split(":")[0]
            val = txt[len(key)+1:].strip()
            txts[key] = val
    txt_dict[d] = txts

# sanity check
for k, v in txt_dict.items():
    expect = int(re.findall(r'\d+', k)[0])  # dir name is like XXX300 (contains 300 wavs)
    assert expect == len(v.keys())

# split train/dev/eval sets
set_ratio = dict(train=0.8, dev=0.1, eval=0.1)
txt_sets = dict(train=[], dev=[], eval=[])
for k, v in txt_dict.items():
    vk = sorted(v.keys())
    pair = ["{} {}\n".format(vkk, v[vkk]) for vkk in vk]
    n = len(pair)
    sets = dict()
    ntrain = int(set_ratio["train"] * n)
    neval = int(set_ratio["eval"] * n)
    txt_sets["train"] += pair[:ntrain]
    txt_sets["eval"] += pair[ntrain:ntrain+neval]
    txt_sets["dev"] += pair[ntrain+neval:]


# write scp
for k, v in txt_sets.items():
    os.makedirs("data/" + k, exist_ok=True)
    with open("data/" + k + "/text", "w") as f:
        f.writelines(sorted(v))
