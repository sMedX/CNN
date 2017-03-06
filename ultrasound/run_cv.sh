#! /bin/sh -ex

for mask in aug242015205152 aug262015184947 aug272015194332 aug272015194427 aug312015195035 aug312015195117 aug312015195204 aug312015195255;
do
    rm -rf debug

    mkdir debug
    python3 train.py --segmentation_dataset_validation_set_regex="$mask"'.*' >debug/log 2>debug/log
    mv debug debug.$mask
done
