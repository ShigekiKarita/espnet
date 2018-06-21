# ESPnetで新しいデータセットを使うには

## 概要

[ESPnet: End-to-End Speech Processing Toolkit](https://arxiv.org/abs/1804.00015) は最先端の音声認識ツールキットです．ESPnetは**End-to-End**なので，[従来のツールキット]((http://kaldi-asr.org/doc/data_prep.html))よりもずっと簡単に自分で音声/テキストペアを追加して，学習・認識させることができます．このチュートリアルでは実用的な例として，最近できたばかりの日本語音声のデータセットである[JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)コーパスを整備して，ESPnetによる音声認識をやってみようと思います.

最終的な音声認識スクリプトはこちらです https://github.com/ShigekiKarita/espnet/tree/jsut/egs/just/asr1/run.sh

## はじめに

ESPnetによる音声認識システムを作る前に，準備すべきデータは
- 音声: 発話単位で区切られており，WAVに変換可能なもの (ffmpegなどで)
- テキスト： 上記の音声と対応づいたもの

このチュートリアルでは，上記のファイルを既存の音声認識ツールキットKaldiにおけるSCP形式 (スペース区切りのキー・値のペアファイル) として整形します．なぜならESPnetの入出力は[Kaldi互換の機構](http://kaldi-asr.org/doc/io.html)を採用しているからです．ここでキー（または発話ID）は同じ音声・テキストの値を結びつけるものです．またwav.scpにおける値とは，任意のシェルコマンド(ただし標準出力にパイプして`|`で終わる文)か，WAVファイルへのパスです．

- data/train/wav.scp. 
```
FILE001 ffmpeg -i /foo/bar.mp4 -ss 3.0 -t 1.0 -f wav -acodec pcm_s16le -ar 16000 -ac 1 - | 
FILE002 /foo/bar2.wav
```

WAVファイルの作成方法として，私のオススメはffmpegを使うことです．なぜならあらゆるファイルからWAVを変換・抽出できるためです `ffmpeg -i <file-path> -ss <start-time-sec> -t <duration-sec> -f wav -acodec pcm_s16le -ar 16000 -ac 1`.

- data/train/text
```
FILE001 hi, can you hear me?
FILE002 ESPnet is nice!
```

## step 1: 必要なもの

### ESPnet

``` bash
git clone https://github.com/ShigekiKarita/espnet -b jsut
cd espnet/tools
make -j6 -f conda.mk PYTHON_VERSION=3.6
```

Kaldiのインストールには非常に時間がかかるでしょう．Makeはあらゆるものをインストールしようとしますが，このチュートリアルではvirtualenv, chainer, pytorch および kaldi が必要なだけです．Makefileやconda.mkの中身を見て，それぞれ個別にインストールしてもかまいません．諦めないでください．

ESPnetのインストールが終われば，とても小さいAN4データセットを試してみましょう．
``` console
$ cd egs/an4/asr1
$ CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --backend pytorch
```
もしGPUをもっていなければ`$ ./run.sh`として動かしましょう.

学習の経過ログは `tail -f ./exp/train_nodev_*/train.log`として確認できます．
最終的に文字誤り率 (CER) 18.2% が数分後に達成されるはずです.悪くない結果です！

ここで，現在のディレクトリは以下のようになっているでしょう．
``` console
$ tree  -L 1
.
├── RESULTS     # 参考の結果
├── cmd.sh      # ジョブスケジューラの設定
├── conf        # 設定ファイル置き場
├── data        # 整形されたデータセット　（チュートリアルではこれを作ります）
├── downloads   # 元データ
├── dump        # 一時的に置かれたファイル
├── exp         # 実験ディレクトリ (pythonスクリプトのログや結果などが置かれます)
├── fbank       # 前処理された音声特徴量置き場
├── local       # タスク依存のスクリプト置き場
├── path.sh     # 実行ファイルパスの設定
├── run.sh      # メインのスクリプト
├── steps -> ../../../tools/kaldi/egs/wsj/s5/steps
└── utils -> ../../../tools/kaldi/egs/wsj/s5/utils
```

今回のチュートリアルは，未加工のデータセットを整形するスクリプトをlocal/以下に書いてdata/ディレクトリを作って学習・認識を動かすのがゴールです．
最終的なdata/ディレクトリはAN4と同様に以下のようになるでしょう．
``` console
$ tree data/ -L2                                                                                                                       
├── lang_1char                                                                                                                    
│   └── train_nodev_units.txt   # データセット中の文字のリスト・辞書
├── test                        # 評価データ置き場
│   ├── feats.scp               # 発話IDつきの音声特徴量ファイルのリスト (自動生成)
│   ├── spk2utt                 # 発話IDつきの話者IDリスト (必要に応じて使う)
│   ├── split8utt               # 並列で認識を実行するために分割した発話リスト (自動生成)
│   ├── text                    # 発話IDつきのテキストのリスト (NOTE: これを作ります！)
│   ├── utt2spk                 # 話者IDつきの発話IDリスト (必要に応じて使う)
│   └── wav.scp                 # 発話IDつきの音声WAVファイのリスト (NOTE: これを作ります！)
├── train                       (test/と同様の中身) 学習データ置き場
├── train_dev                   (test/と同様の中身) 交差検証のための開発データ置き場
└── train_nodev                 (test/と同様の中身) 開発データを除いた学習データ置き場
```

### ffmpeg

ffmpegも忘れずインストールしましょう

``` bash
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar -xvf ffmpeg-release-64bit-static.tar.xz
export PATH=`pwd`/ffmpeg-3.4-64bit-static:$PATH
```

## step2: overview run.sh

[CSJ](http://pj.ninjal.ac.jp/corpus_center/csj/)ディレクトリをJSUTの下書きとしてコピーします．

``` console
$ cd espnet/egs
$ cp -r csj just
$ cd just/asr1
```

というのも既存のレシピはときどき言語・タスク依存の前処理を含んでいるので，最初のもとになるスクリプトとして便利です．
- WSJ ： 英語の読み上げ音声むき
- Tedlium : 英語の話し音声むき
- Switchboard : 英語の会話音声むき
- HKUST : 中国語むき
- CSJ : 日本語の話し音声むき
- Voxforge : ヨーロッパ語，多言語音声むき
- CHiME4/5 : 雑音下の多チャネル音声むき

それでは run.sh をJSUTデータセット向けに書き換えましょう．

``` bash
#!/bin/bash

# NOTE: espnet/tools/ 以下にPATHを通す設定読み込み(例: virtualenv, kaldi/src/featbin)
. ./path.sh 
# NOTE: ジョブスケジューラの設定読み込み(例: queue.pl, slurm.pl,run.pl)
. ./cmd.sh

# NOTE: 沢山の学習・認識用のコマンドラインオプション
...
. utils/parse_options.sh || exit 1;

# NOTE: ここを書き換えます！
if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data Preparation"
    ...
fi

# NOTE: ここを書き換えます！
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    ...
fi

# NOTE: ここは書き換えません
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    ...
fi

# NOTE: ここの言語モデル学習は簡単のため消します
# You can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=exp/train_rnnlm_2layer_bs256
mkdir -p ${lmexpdir}
if [ ${stage} -le 3 ]; then
    echo "stage 3: LM Preparation"
    ...
fi

# NOTE: ここは書き換えません
if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    ...
fi

# NOTE: ここは --rnnlm and --lm-weight オプションを簡単のため消します
if [ ${stage} -le 5 ]; then     
    echo "stage 5: Decoding"    
    ...
fi
```

### コメント
- Stage 1: 音声認識ではニューラルネットワークに入力する音声特徴量として log Mel filterbank (通称 FBANK) を使います. 詳細は[こちら](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) for detail
- Stage 2: ESPneｔは .json に学習や評価に使う情報(例：話者，テキスト，長さ，次元など)を格納し，それと別に音声特徴量をkaldi形式で作ります.また辞書とはデータセット中の文字を一覧にしたものです. 
- Stage 3: ESPnetは高精度な音声認識のため言語モデルを統合可能ですが，簡単のため今回は扱いません．
- Stage 4: ニューラルネットワークの学習です．Pytorchバックエンドが高速でオススメです．　`espnet/src/nets/e2e_asr_attctc_th.py`の`class E2E(torch.nn.Module)`を改造することで，自分の考えたニューラルネットワークを使えます．
- Stage 5: 音声の認識や評価を行います.ここでも言語モデルの統合はオフにしています．

## step 3.  run.shを書き換える

### stage 0. JSUT zipのダウンロード

何も難しくありません．普通のシェルスクリプトです．
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
fi
```

つぎにすべきことは
- データセットを train/dev/eval に分割
- data/xxx/wav.scp を分割したセット毎に作る
- data/xxx/text を分割したセット毎に作る

残念ながら，JSUTは公式にtrain/dev/evalセットを提供していないので，各ディレクトリのファイルをソートして，上位80/10/10%をそれぞれのセットとして使います．

### stage 0. textの準備

textファイルはとても簡単です．なぜならすでにscpファイルのような形をしているからです．

- ファイル: downloads/jsut_ver1.1/basic5000/transcript_utf8.txt
- 中身: 
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
日本語なのに形態素解析や単語区切りがなくても大丈夫です．ESPnetは文字レベルの音声認識を得意としています．

ここで，Python3を使って整形スクリプトを作ります．
- [local/jsut_prepare_text.py](local/jsut_prepare_text.py)


``` python
#!/usr/bin/env python3
import re
from glob import glob
import os

# ファイル名の例: ..../downloads/jsut_ver1.1/basic5000/transcript_utf8.txt
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

# ファイル数のチェック
for k, v in txt_dict.items():
    expect = int(re.findall(r'\d+', k)[0])  # XXX300のようなディレクトリは300発話含む
    assert expect == len(v.keys())

# train/dev/eval セットの分割
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


# scp形式の書き込み
for k, v in txt_sets.items():
    os.makedirs("data/" + k, exist_ok=True)
    with open("data/" + k + "/text", "w") as f:
        f.writelines(sorted(v))
```
NOTE: scp形式のファイル中で発話IDはソート済みでなくてはなりません．

### stage 0. prepare wav.scp

Waveファイルはすでに発話単位でユニークな分割がされていました（そのため発話IDとしてファイル名を使っています）．
しかしながら，元ファイルの48kHzは音声認識用としては大きすぎますので，ffmpegを受かって16kHzにダウンサンプリングします．
- [local/jsut_prepare_wav.py](local/jsut_prepare_wav.py)

``` python
#!/usr/bin/env python3
import re
from glob import glob
import os

# ファイル名の例: ..../downloads/jsut_ver1.1/utparaphrase512/wav/UT-PARAPHRASE-sent254-phrase2.wav
wav_list = glob("downloads/jsut_ver1.1/*/wav/*.wav")
wav_dict = dict()
for wav in wav_list:
    k = wav.split("/")[2]
    if k not in wav_dict:
        wav_dict[k] = []
    v = os.getcwd() + "/" + wav
    wav_dict[k].append(v)

# ファイル数のチェック
for k, v in wav_dict.items():
    expect = int(re.findall(r'\d+', k)[0])  # XXX300のようなディレクトリは300発話含む
    assert expect == len(v)

# train/dev/eval セットの分割
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

# scp形式の書き込み
n_ext = len(".wav")
for k, v in wav_sets.items():
    os.makedirs("data/" + k, exist_ok=True)
    with open("data/" + k + "/wav.scp", "w") as w, \
         open("data/" + k + "/spk2utt", "w") as s, \
         open("data/" + k + "/utt2spk", "w") as u:
        for wav in sorted(v):
            uttid = os.path.basename(wav)[:-n_ext]
            # ffmpeg によるダウンサンプリング
            w.write("{} ffmpeg -i {} -f wav -acodec pcm_s16le -ar 16000 -ac 1 - |\n".format(uttid, wav))
            s.write("{} {}\n".format(uttid, uttid))
            u.write("{} {}\n".format(uttid, uttid))
```

NOTE: JSUTは話者情報を持っていないので `spk2utt` と `utt2spk`ファイルには適当に発話IDを話者IDとしていれておきました.話者情報は話者適応を行う際に便利ですが，ESPnetにはいまのところ話者適応を行う実験レシピはありません．

ようやく，最も難しい部分は終わりました！

### stage 1. FBANK の生成

ほとんど何も書き換える必要はありません．ここではただ3つのコマンドを使うだけです．
1. `steps/make_fbank_pitch.sh` :  wav.scp　から fbankのリスト (feat.scp) を作ります
2. `compute-cmvn-stats` : ニューラルネットワークの入力を正規化するための統計量を計算して結果をファイルcmvn.arkに出力します
3. `dump.sh` : 一時jsonファイルに音声特徴量以外の情報をまとめます．

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

残りの部分はほとんど何もしていない (使わない部分をコメントアウトしただけ)　ので実際のスクリプトで確認してください https://github.com/ShigekiKarita/espnet/blob/jsut/egs/just/asr1/run.sh

これで終わりです！

## step 3. 全てを動かす

動かし方は全て an4/asr1/run.sh　のときと同じです．

``` console
CUDA_VISIBLE_DEVICES=0 ./run.sh --ngpu 1 --backend pytorch --etype blstmp
```
私の環境 (gtx1080ti) では実行に40分くらいかかりました.以下が認識精度の経過グラフです．

![acc](./acc.png)

うーん，これは学習を止めるタイミングが速すぎたようです. (こういった実験結果やログは exp/train_xxx　以下にあります)

## step 4. extend this recipe

前節でみたように，この実験結果は十分なものではありません．あなた自身で run.sh をチューニングして認識精度を上げましょう！例えば...
- もっと長いエポック数を回す．例`$ ./run.sh --epochs 50`． さきほどの実験結果は15 epochでしたが収束していないので
- 最適化パラメータ(アルゴリズムやbatchsizeなど)を弄る
- ニューラルネットワークの設定(unit数やレイヤーの種類など)を弄る
- デコードのパラメータ(スコア重みなど)を弄る
- 言語モデルを有効にする
- データセットが比較的小さいのでdata augumentation を行う (再生スピードの変化など ffmpeg のオプションを調べましょう) 

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

In this result, the current system achieved character error rate 25.1%!. And you can see some kanji conversion errors. Therefore language model (LM) could improve these errors?

[このリポジトリ](https://github.com/ShigekiKarita/espnet/tree/jsut/egs/just/asr1) をフォークして楽しみましょう！ もし良い結果が得られたらPRしてください＾＾

もし質問があれば,[@kari_tech](https://twitter.com/kari_tech)まで
