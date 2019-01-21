#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false

# network archtecture
ninit=pytorch
ntype=transformer

# encoder related
input_layer=conv2d
elayers=6
eunits=1024
eprojs=320
subsample=0 # skip every n frame from input to nth layers
# decoder related
dlayers=6
dunits=1024
# attention related
aheads=4
adim=256

# hybrid CTC/attention
mtlalpha=0.0

# label smoothing
lsm_weight=0.1

# minibatch related
batchsize=16
maxlen_in=512  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
len_norm=True
opt=noam
epochs=100
lr_init=10.0
warmup_steps=25000
dropout=0.1
attn_dropout=0.0
accum_grad=2
grad_clip=5

# rnnlm related TODO(karita)
# use_wordlm=true     # false means to train/use a character LM
# lm_vocabsize=65000  # effective only for word LMs
# lm_layers=1         # 2 for character LMs
# lm_units=1000       # 650 for character LMs
# lm_opt=sgd          # adam for character LMs
# lm_batchsize=300    # 1024 for character LMs
# lm_epochs=20        # number of epochs
# lm_maxlen=40        # 150 for character LMs
# lm_resume=          # specify a snapshot file to resume LM training
# lmtag=              # tag for managing LMs

# decoding parameter
lm_weight=0.0
beam_size=1
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.0
n_average=10
recog_model=model.avg.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# scheduled sampling option
samp_prob=0.0

# data
wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
train_dev=test_dev93
train_test=test_eval92
recog_set="test_dev93 test_eval92"

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train_si284 test_dev93 test_eval92; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/wsj/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{10,11,12,13}/${USER}/espnet-data/egs/wsj/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# # It takes about one day. If you just want to do end-to-end ASR without LM,
# # you can skip this and remove --rnnlm option in the recognition (stage 5)
# if [ -z ${lmtag} ]; then
#     lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
#     if [ $use_wordlm = true ]; then
#         lmtag=${lmtag}_word${lm_vocabsize}
#     fi
# fi
# lmexpdir=exp/train_rnnlm_${backend}_${lmtag}
# mkdir -p ${lmexpdir}

# TODO(karita)
# if [ ${stage} -le 3 ]; then
#     echo "stage 3: LM Preparation"
    
#     if [ $use_wordlm = true ]; then
#         lmdatadir=data/local/wordlm_train
#         lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
#         mkdir -p ${lmdatadir}
#         cat data/${train_set}/text | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
#         zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
#                 | grep -v "<" | tr [a-z] [A-Z] > ${lmdatadir}/train_others.txt
#         cat data/${train_dev}/text | cut -f 2- -d" " > ${lmdatadir}/valid.txt
#         cat data/${train_test}/text | cut -f 2- -d" " > ${lmdatadir}/test.txt
#         cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
#         text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
#     else
#         lmdatadir=data/local/lm_train
#         lmdict=$dict
#         mkdir -p ${lmdatadir}
#         text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
#             | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
#         zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
#             | grep -v "<" | tr [a-z] [A-Z] \
#             | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
#         text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text \
#             | cut -f 2- -d" " > ${lmdatadir}/valid.txt
#         text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_test}/text \
#                 | cut -f 2- -d" " > ${lmdatadir}/test.txt
#         cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
#     fi

#     # use only 1 gpu
#     if [ ${ngpu} -gt 1 ]; then
#         echo "LM training does not support multi-gpu. signle gpu will be used."
#     fi
#     ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
#         lm_train.py \
#         --ngpu ${ngpu} \
#         --backend ${backend} \
#         --verbose 1 \
#         --outdir ${lmexpdir} \
#         --train-label ${lmdatadir}/train.txt \
#         --valid-label ${lmdatadir}/valid.txt \
#         --test-label ${lmdatadir}/test.txt \
#         --resume ${lm_resume} \
#         --layer ${lm_layers} \
#         --unit ${lm_units} \
#         --opt ${lm_opt} \
#         --batchsize ${lm_batchsize} \
#         --epoch ${lm_epochs} \
#         --maxlen ${lm_maxlen} \
#         --dict ${lmdict}
# fi


if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${backend}_${ntype}_${input_layer}_e${elayers}_subsample${subsample}_unit${eunits}_d${dlayers}_unit${dunits}_aheads${aheads}_adim${adim}_mtlalpha${mtlalpha}_${opt}_clip${grad_clip}_sampprob${samp_prob}_ngpu${ngpu}_bs${batchsize}_lr${lr_init}_warmup${warmup_steps}_dropout${dropout}_attndropout${attn_dropout}_mli${maxlen_in}_mlo${maxlen_out}_ninit_${ninit}_epochs${epochs}_accum${accum_grad}_lennorm${len_norm}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${backend}_${tag}
fi

mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --accum-grad ${accum_grad} \
        --ngpu ${ngpu} \
        --ntype ${ntype} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --aheads ${aheads} \
        --adim ${adim} \
        --mtlalpha ${mtlalpha} \
        --batch-size ${batchsize} \
        --dropout-rate ${dropout} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --grad-clip ${grad_clip} \
        --sampling-probability ${samp_prob} \
        --epochs ${epochs} \
        --transformer-lr ${lr_init} \
        --transformer-warmup-steps ${warmup_steps} \
        --transformer-input-layer ${input_layer} \
        --transformer-attn-dropout-rate ${attn_dropout} \
        --transformer-length-normalized-loss ${len_norm} \
        --transformer-init ${ninit} \
        --lsm-weight ${lsm_weight}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=3

    if [ ! -e ${expdir}/results/${recog_model} ]; then
        average_checkpoints.py --snapshots ${expdir}/results/snapshot.ep.* --out ${expdir}/results/${recog_model} --num ${n_average}
    fi

    # TODO(karita) CTC and LM joint decoding
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio} #_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
        if [ $use_wordlm = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        if [ $lm_weight == 0 ]; then
            recog_opts=""
        fi
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --lm-weight ${lm_weight} \
            $recog_opts &
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi
