s = ""
for lm in [0.3, 0.5, 0.7, 0.9]:
    ctc_results = []
    for ctc in [0.1, 0.3, 0.5, 0.7]:
        results = []
        for setname in ["dev_clean", "dev_other", "test_clean", "test_other"]:
            path = f"./exp/train_960_pytorch_nbpe5000_ngpu8_train_pytorch_transformer.v2.seqs_specaug/decode_sbatch_{setname}_decode_pytorch_transformer_beam-size10_ctc-weight{ctc}_lm-weight{lm}_/result.wrd.txt"
            wer = "fail"
            try:
                with open(path, "r") as f:
                    for line in f:
                        if "Avg" in line:
                            wer = line.split()[-3]
                            break
            except FileNotFoundError:
                pass
            results.append(wer)
        ctc_results.append(" / ".join(results))
    s += "    &    ".join(ctc_results) + "\n"
print(s)
