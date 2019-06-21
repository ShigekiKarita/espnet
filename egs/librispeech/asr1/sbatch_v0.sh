PATH=../../../utils:$PATH

for el in 24 36 48; do
    ./run.sh --ngpu 2 \
             --nbpe 16000 \
             --stage 4 \
             --decode-config conf/tuning/decode_pytorch_transformer.yaml \
             --train-config $(change_yaml.py conf/tuning/train_pytorch_transformer.yaml -a elayers=$el) &
    sleep 2
done

for u in 2560 3072; do
    ./run.sh --ngpu 4 \
             --nbpe 16000 \
             --stage 4 \
             --decode-config conf/tuning/decode_pytorch_transformer.yaml \
             --train-config $(change_yaml.py conf/tuning/train_pytorch_transformer.yaml -a elayers=$el) &
    sleep 2
done

wait

