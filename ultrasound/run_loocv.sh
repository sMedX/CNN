#! /bin/sh -ex

for mask in `ls ~/datasets/ultrasound_mw/mask | sed 's|_.*||' | sort | uniq`;
do
    rm -rf debug

    if [ -d "debug.$mask" ]; then
	echo "skipping $mask as debug.$mask already exists"
	continue
    fi

    echo "validation set is $mask.*"
    mkdir debug
    python3 train.py --segmentation_dataset_validation_set_regex="$mask"'.*' >debug/log 2>debug/log
    mv debug debug.$mask
done

tail -n 20 debug*/log | grep '^tp = ' | sed 's|,||g' | \
    awk '{ tp = tp + $3; fp = fp + $6; tn = tn + $9; fn = fn + $12 } END { P = tp / (tp + fp); R = tp / (tp + fn); F1 = 2 * P * R / (P + R); print("tp =", tp, "fp =", fp, "tn =", tn, "fn =", fn); print("P =", P, "R =", R, "F1 =", F1);  }'
