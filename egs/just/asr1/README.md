# How to prepare dataset in ESPnet

[(Japanese)](README.ja.md)

## abstract

[ESPnet: End-to-End Speech Processing Toolkit](https://arxiv.org/abs/1804.00015) is a state-of-the-art speech recognition toolkit. As ESPnet is **end-to-end**, you can train/recognize your own speech/text pairs more easily than [other toolkits](http://kaldi-asr.org/doc/data_prep.html). This tutorial describes how to prepare your own dataset in ESPnet in a very simple way. For practical example, we gonna format a brand new corpus [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut).

You can find the final speech recogntion scripts at https://github.com/ShigekiKarita/espnet/tree/jsut/egs/just/asr1/run.sh

## introduction

Before building ESPnet speech-to-text system, you need to prepare
- speech: splitted as utterances, convertable to wav (via ffmpeg)
- text: able to pair with speech utterances

In this tutorial, these files are formatted into a kaldi's SCP style (key/value pairs) because ESPnet depends on [kaldi's IO mechanism](http://kaldi-asr.org/doc/io.html). Note that key (i.e., utt-id) should be matched between same speech and text values. The value of speech (wav.scp) can be any shell commands that pipe into stdout and end with `|`) or path to wav files like this.

- data/train/wav.scp. 
```
FILE001 ffmpeg -i /foo/bar.mp4 -ss 3.0 -t 1.0 -f wav -acodec pcm_s16le -ar 16000 -ac 1 - | 
FILE002 /foo/bar2.wav
```

To create wav files, I recommend you to use [ffmpeg](https://johnvansickle.com/ffmpeg/) because it can convert and extract any audio files like `ffmpeg -i <file-path> -ss <start-time-sec> -t <duration-sec> -f wav -acodec pcm_s16le -ar 16000 -ac 1`.

- data/train/text
```
FILE001 hi, can you hear me?
FILE002 ESPnet is nice!
```

## step 1: get requirements

### ESPnet

``` bash
git clone https://github.com/ShigekiKarita/espnet -b jsut
cd espnet/tools
make -j6 -f conda.mk PYTHON_VERSION=3.6
```
It takes very long time in kaldi build (around 15min). Make tries to install everything for ESPnet but we only need virtualenv, chainer and kaldi here. You can manually install them by looking inside Makefile and conda.mk. do not give up. Once you finished installing ESPnet, you can test a small free dataset AN4.

``` console
$ cd egs/an4/asr1
$ CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --backend pytorch --etype blstmp
```
if you do not have GPUs, use `$ ./run.sh --backend pytorch --etype blstmp`. I recommend you to use pytorch backend instead of chainer becuase it is much faster.

You can check the training reports by  `tail -f ./exp/train_nodev_*/train.log`.
Finally, it would result in character-error-rate (CER) 18.2% in several minutes. Not so bad!

Now you find that current dir is organized like
``` console
$ tree  -L 1
.
├── RESULTS     # reference result
├── cmd.sh      # job scheduler settings
├── conf        # config files
├── data        # formatted dataset files (<- this is what we forcus on)
├── downloads   # downloaded raw files
├── dump        # temp dumped json files
├── exp         # experiment dir (running python scripts here)
├── fbank       # preprocessed input speech features
├── local       # task (dataset) specific scripts for formatting
├── path.sh     # path settings for espnet/tools/*
├── run.sh      # main script
├── steps -> ../../../tools/kaldi/egs/wsj/s5/steps
└── utils -> ../../../tools/kaldi/egs/wsj/s5/utils
```

In this tutorial, we aim to format the raw dataset into `data/` dir by making `local/` scripts.
The final data/ should be organized as same as an4

``` console
$ tree data/ -L2                                                                                                                       
├── lang_1char                                                                                                                    
│   └── train_nodev_units.txt   # list or dictionary of characters
├── test                        # evaluation data
│   ├── feats.scp               # fbank lists with utt-id (auto generated)
│   ├── spk2utt                 # speaker lists with utt-id (optional)
│   ├── split8utt               # split dataset for parallel decoding (auto generated)
│   ├── text                    # text lists with utt-id (NOTE: we need to create)
│   ├── utt2spk                 # utt lists with speaker-id (optional)
│   └── wav.scp                 # speech wav lists with utt-id (NOTE: we need to create)
├── train                       (similar to test/) training data
├── train_dev                   (similar to test/) cross-validation data for monitoring 
└── train_nodev                 (similar to test/) training data without cross-validation data
```

### ffmpeg

do not forget to install ffmpeg

``` bash
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar -xvf ffmpeg-release-64bit-static.tar.xz
export PATH=`pwd`/ffmpeg-3.4-64bit-static:$PATH
```

## step2: overview run.sh

Let's copy CSJ](http://pj.ninjal.ac.jp/corpus_center/csj/) dir as JSUT dir. 

``` console
$ cd espnet/egs
$ cp -r csj just
$ cd just/asr1
```

As the existing recipes sometimes contain  some language/task specific preprocessing, I recommend them for a initial script
- WSJ for reading English
- Tedlium for English talk
- Switchboard for English conversation
- HKUST for Chinese
- CSJ for Japanese
- Voxforge for European or mixed languages
- CHiME4/5 for noisy multi-channel speech

now edit run.sh that contains the following steps

``` bash
#!/bin/bash

# NOTE: setup path of espnet/tools/ (e.g., virtualenv, kaldi/src/featbin)
. ./path.sh 
# NOTE: setup job scheduler (queue.pl or slurm.pl) or local jub runner (run.pl)
. ./cmd.sh

# NOTE: lots of training and decoding configs as cmd options (discussed later)
...
. utils/parse_options.sh || exit 1;

# NOTE: WE NEED TO REWRITE HERE
if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    ...
fi

# NOTE: WE NEED TO REWRITE HERE
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    ...
fi

# NOTE: we do nothing here
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    ...
fi

# NOTE: we will remove this stage for simplicity
# You can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=exp/train_rnnlm_2layer_bs256
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    ...
fi

# NOTE: we do nothing here
if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    ...
fi

# NOTE: we need to remove --rnnlm and --lm-weight options here
if [ ${stage} -le 5 ]; then     
    echo "stage 5: Decoding"    
    ...
fi
```

### comments
- Stage 1: In speech recognition, we use log Mel filterbank (a.k.a. FBANK) feature basically. see [this page](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) for detail
- Stage 2: ESPnet uses .json files to store training/evaluation informations (e.g., speaker, text, length, dim) except for speech features that stored as a kaldi format. The dictionary is just a list of all the characters appeared in the dataset. 
- Stage 3: ESPnet also can integrate with language models during decoding to achieve higher accuracy but you can avoid it.
- Stage 4: The training part of this recipe. I recommend to use pytorch backend because it is much faster. You can customize your models by rewriting `class E2E(torch.nn.Module)` in `espnet/src/nets/e2e_asr_attctc_th.py`
- Stage 5: The evaluation part of this recipe. We omit language model scoring options here for simplicity.

## step 3. rewriting run.sh

### stage 0. download JSUT zip

nothing is difficult.

``` bash
# data
dl_dir=./downloads
jsut_root=${dl_dir}/jsut_ver1.1
data_url=http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip

if [ ${stage} -le 0 ]; then
    echo "stage 0: Data Preparation"
    mkdir -p ${dl_dir}
    if [ ! -d ${jsut_root} ]; then
        cd ${dl_dir}
        wget $data_url
        unzip ${jsut_root}.zip
        cd -
    fi
    # TODO 
    local/jsut_prepare_wav.py
    local/jsut_prepare_text.py
    for u in data/*/utt2spk; do
        utils/utt2spk_to_spk2utt.pl $u > $(dirname $u)/spk2utt
    done
fi
```

Next, we need to
- split one dataset into train/dev/eval sets
- create data/xxx/wav.scp for each sets
- create data/xxx/text for each sets

Unfortunately, JSUT does not provide official train/dev/eval sets.
Here we decide to split them into 80/10/10 % for each dirs in sorted order.

### stage 0. prepare text

text is much easier than wav.scp because it already looks like scp

- txt path : downloads/jsut_ver1.1/basic5000/transcript_utf8.txt
- txt content: 
```
BASIC5000_0001:水をマレーシアから買わなくてはならないのです。
BASIC5000_0002:木曜日、停戦会談は、何の進展もないまま終了しました。
BASIC5000_0003:上院議員は私がデータをゆがめたと告発した。
BASIC5000_0004:１週間して、そのニュースは本当になった。
BASIC5000_0005:血圧は、健康のパロメーターとして重要である。
BASIC5000_0006:週に四回、フランスの授業があります。
BASIC5000_0007:許可書がなければここへは入れない。
BASIC5000_0008:大声で泣きながら、女の子は母親を探していた。
BASIC5000_0009:無罪の人々は、もちろん放免された。
BASIC5000_0010:末期試験に備えて、本当に気合いを入れて勉強しなきゃ。
...
```
Do not afraid if there are no word boundaries, ESPnet is good at character-level speech recognition.

Here I use python3 because it handles UTF8 texts nicely.

- [local/jsut_prepare_text.py](local/jsut_prepare_text.py)


``` python
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
```
NOTE: you need to sorted uttid in scp files.

### stage 0. prepare wav.scp

Wave files are already separated as unique utterances (and filename can be directly used as uttrance-id) but its sampling rate is 48kHz (good for TTS). We downsample it to 16kHz by ffmpeg because it is too much for speech recognition. I also use python3 to format scp files as follows

- [local/jsut_prepare_wav.py](local/jsut_prepare_wav.py)

``` python
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
            u.write("{} {}\n".format(uttid, uttid))
```

NOTE: As JSUT does not provide speaker information, we simply put dirname as speaker-id.

now the most difficult part is **finished!**

### stage 1. JSON and FBANK generation

almost nothing to rewrite. just calling 3 commands
1. `steps/make_fbank_pitch.sh` : create fbank (feat.scp) from wav.scp
2. `compute-cmvn-stats` : create cepstram-mean-variance (cmvn.ark) to normalize fbank inputs of neural network
3. `dump.sh` : create temp json files with text (transcription) informations

``` bash
train_set=train
train_dev=dev
recog_set="dev eval"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    for x in ${train_set} ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
        
    # dump features for evaluation
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi
```

dumped json `data/train/deltafalse/data.json` is organized as follows
```
{
    "utts": {
        "BASIC5000_1698": {
            "utt2spk": "basic5000", 
            "input": [
                {
                    "shape": [
                        384, 
                        83
                    ], 
                    "feat": "/home/skarita/work/dev/espnet/egs/just/asr1/dump/train/deltafalse/feats.2.ark:5174113", 
                    "name": "input1"
                }
            ], 
            "output": [
                {
                    "text": "僕の片言の英語が、なんとか通じたのでほっとしました。", 
                    "shape": [
                        26, 
                        2561
                    ], 
                    "name": "target1", 
                    "token": "僕 の 片 言 の 英 語 が 、 な ん と か 通 じ た の で ほ っ と し ま し た 。", 
                    "tokenid": "2157 85 2086 777 85 289 810 52 19 81 119 79 51 1684 64 71 85 78 98 74 79 63 101 63 71 20" 
                }
            ]
        }, 
        "BASIC5000_1699": {
            "utt2spk": "basic5000", 
            "input": [
...
```

See https://github.com/ShigekiKarita/espnet/blob/jsut/egs/just/asr1/run.sh for the remaining part (only small changes from csj/asr1/run.sh).

We have done!

## step 3. run them all

Everything is same to an4/asr1/run.sh

``` console
CUDA_VISIBLE_DEVICES=0 ./run.sh --backend pytorch --etype blstmp
```
in my environment (gtx1080ti), it took 40 mins.

![acc](./acc.png)

hmm, it might be too early to stop. (you can find some experiment results like this in exp/train_xxx)

## step 4. extend this recipe

As seen in the previous section, the result is not great enough. You can tune up this `run.sh` as follows
- run more epochs e.g., `$ ./run.sh --epochs 50` because it does not seem to be converged in 15 epochs
- tweak optimizer parameters
- tweak neural networks
- tweak decoding parameters
- enable language modeling
- apply data augumentation techniques (e.g., speed perturbation) because this dataset is small

## step 5. error analysis

when I changed epochs from 30 to 50, it works well.

You can confirm ESPnet's recognition results in `exp/train_xxx/decode_yyy/results.txt` like

```

...
| SPKR                          | # Snt  # Wrd | Corr     Sub    Del     Ins    Err   S.Err |
....
| Sum/Avg                       |  773   18869 | 78.0    19.5    2.5     3.1   25.1    96.2 |

...
Speaker sentences   0:  ut_paraphrase_sent291_phrase2   #utts: 1
id: (ut_paraphrase_sent291_phrase2-ut-paraphrase-sent291-phrase2)
Scores: (#C #S #D #I) 19 3 1 0
REF:  前 よ り 面 倒 だ が 、 詳 細 が よ く 分 か る よ う に な っ た 。 
HYP:  前 よ り 面 道 だ が 、 少 年 が よ く 分 *** る よ う に な っ た 。 
Eval:                 S               S   S                   D                                   

Speaker sentences   1:  ut_paraphrase_sent291_phrase1   #utts: 1
id: (ut_paraphrase_sent291_phrase1-ut-paraphrase-sent291-phrase1)
Scores: (#C #S #D #I) 20 3 0 0
REF:  前 よ り 面 倒 だ が 、 明 細 が よ く 分 か る よ う に な っ た 。 
HYP:  前 よ り 面 道 だ が 、 名 線 が よ く 分 か る よ う に な っ た 。 
Eval:                 S               S   S                                                       
...
```

In this result, the current system achieved character error rate 25.1%!. However, you can see some kanji conversion errors. Therefore language model (LM) could reduce these errors?

## step 6. fork me 

Finally you have mastered ESPnet. Fork [this](https://github.com/ShigekiKarita/espnet/tree/jsut/egs/just/asr1) and have fun! Then, give us PR if you got nice results:-)

If you have questions, [you can ask here](https://github.com/espnet/espnet/issues).
