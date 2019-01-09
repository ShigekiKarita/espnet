for name in dev eval; do
    echo "#${name}"
    for d in "$1/decode_test_${name}*/result.txt" "$1/decode_test_${name}*/result.wrd.txt"; do
        echo $d
        grep -e Avg -e SPKR -m 2 $d
    done
done
