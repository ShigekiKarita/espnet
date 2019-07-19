import logging
import json
from glob import glob
topk = 10
score = "validation/main/acc"

def f(nbpe):
    pattern = f"exp/train_960*nbpe{nbpe}*/results/log"

    results = []
    for path in glob(pattern):
        try:
            qpath = path[:-len("/results/log")] + "/q/train.log"
            with open(qpath, "r") as f:
                canceled = any("CANCELLED" in l for l in f)
                if canceled:
                    print(path)
        except IOError:
            continue

f(5000)
f(16000)
