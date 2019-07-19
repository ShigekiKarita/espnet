#!/usr/bin/env bash
asr=./exp/train_960_pytorch_nbpe5000_ngpu8_train_pytorch_transformer.v2.seqs_specaug
lm=./exp/train_rnnlm_pytorch_lm_layer4_unit2048_dropout0.0_batchsize512_unigram5000

. path.sh

# for ctcw in 0.3 0.5 0.7; do
#     for lmw in 0.5 0.7 0.9; do
#         ./decode.sh --nbpe 5000 --nj 16 --expdir $asr --lmexpdir $lm --decode-config $(change_yaml.py conf/tuning/decode_pytorch_transformer.yaml -a beam-size=10 -a ctc-weight=$ctcw -a lm-weight=$lmw) --tag sbatch &
#         sleep 1
#     done
# done

lmw=0.3
ctcw=0.1
./decode.sh --nbpe 5000 --nj 16 --expdir $asr --lmexpdir $lm --decode-config $(change_yaml.py conf/tuning/decode_pytorch_transformer.yaml -a beam-size=10 -a ctc-weight=$ctcw -a lm-weight=$lmw) --tag sbatch &

lmw=0.3
for ctcw in 0.3 0.5 0.7; do
    ./decode.sh --nbpe 5000 --nj 16 --expdir $asr --lmexpdir $lm --decode-config $(change_yaml.py conf/tuning/decode_pytorch_transformer.yaml -a beam-size=10 -a ctc-weight=$ctcw -a lm-weight=$lmw) --tag sbatch &
    sleep 1
done

ctcw=0.1
for lmw in 0.5 0.7 0.9; do
    ./decode.sh --nbpe 5000 --nj 16 --expdir $asr --lmexpdir $lm --decode-config $(change_yaml.py conf/tuning/decode_pytorch_transformer.yaml -a beam-size=10 -a ctc-weight=$ctcw -a lm-weight=$lmw) --tag sbatch &
    sleep 1
done

wait
