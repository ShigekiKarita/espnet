#!/usr/bin/env python3
from datetime import datetime
from glob import glob

def line2time(line):
    return datetime.strptime(line.split(",")[0].strip(), "%Y-%m-%d %H:%M:%S")

def calc(logdir):
    print(logdir)
    result = []
    ilens = []
    for d in glob(logdir + "/decode.*.log"):
        with open(d, "r") as f:
            for line in f:
                if ": input lengths" in line:
                    ilens.append(int(line.split()[-1]))
                if "decoding" in line:
                    start = line2time(line)
                if "prediction" in line:
                    end = line2time(line)
                    s = (end - start).total_seconds()
                    result.append(s)
    print("average [sec/utt]:", sum(result) / len(result))
    import numpy
    print("average [sec/frame]:", numpy.mean(numpy.array(result) / numpy.array(ilens)))

calc("exp/train_960_pytorch_nbpe5000_ngpu8_train_pytorch_transformer.v2.seqs_specaug/decode_avg10_irielm_ep2_beam40_dev_clean_decode_pytorch_transformer_beam-size40_/log")
# calc("exp/train_960_pytorch_nbpe5000_ngpu8_train_pytorch_transformer.shinji_elayers24_aheads8_specaug/decode_ep70_dev_clean_decode_pytorch_transformer_beam-size40_/log")
# calc("exp/train_960_pytorch_nbpe5000_ngpu1_train_pytorch_transformer.shinji_specaug/decode_dev_clean_decode_pytorch_transformer_lm/log")
# calc("exp/train_960_pytorch_nbpe5000_ngpu1_train/decode_dev_clean_decode_lm/log")
calc("exp/train_960_pytorch_nbpe5000_ngpu1_train/decode_tmp_dev_clean_decode_beam-size40_/log")
