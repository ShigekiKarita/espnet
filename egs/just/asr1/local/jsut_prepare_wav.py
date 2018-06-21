#!/usr/bin/env python3
import re
from glob import glob
import os

# ex: ..../downloads/jsut_ver1.1/utparaphrase512/wav/UT-PARAPHRASE-sent254-phrase2.wav
wav_list = glob("downloads/jsut_ver1.1/*/wav/*.wav")
wav_dict = dict()
for wav in wav_list:
    k = wav.split("/")[2]
    if k not in wav_dict:
        wav_dict[k] = []
    v = os.getcwd() + "/" + wav
    wav_dict[k].append(v)

# sanity check
for k, v in wav_dict.items():
    expect = int(re.findall(r'\d+', k)[0])  # dir name is like XXX300 (contains 300 wavs)
    assert expect == len(v)

# split train/dev/eval sets
set_ratio = dict(train=0.8, dev=0.1, eval=0.1)
wav_sets = dict(train=[], dev=[], eval=[])
for k, v in wav_dict.items():
    v = sorted(v)
    n = len(v)
    sets = dict()
    ntrain = int(set_ratio["train"] * n)
    neval = int(set_ratio["eval"] * n)
    wav_sets["train"] += v[:ntrain]
    wav_sets["eval"] += v[ntrain:ntrain+neval]
    wav_sets["dev"] += v[ntrain+neval:]


# write scp
n_ext = len(".wav")
for k, v in wav_sets.items():
    os.makedirs("data/" + k, exist_ok=True)
    with open("data/" + k + "/wav.scp", "w") as w, \
         open("data/" + k + "/utt2spk", "w") as u:
        for wav in sorted(v):
            # use dirname as speaker-id for error analysis
            spkid = wav.split("/")[-3]
            uttid = os.path.basename(wav)[:-n_ext]
            w.write("{} ffmpeg -i {} -f wav -acodec pcm_s16le -ar 16000 -ac 1 - |\n".format(uttid, wav))
            u.write("{} {}\n".format(uttid, spkid))
