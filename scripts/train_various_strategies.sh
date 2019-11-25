#!/usr/bin/env bash

DATASET=$1
BETA=$2
M=$3
K=$4

OUTPUT="$DATASET--vary-strategies"

STRATEGIES="oneshot seq/d:1|e:5 seq/d:1|e:10 seq/d:1|e:20 seq/d:5|e:1 seq/d:10|e:1 seq/d:20|e:1"

for s in $STRATEGIES; do
    time python3 train.py \
        --dataset $DATASET --beta $BETA -M $M \
        --output-dir="./artifacts/$OUTPUT" \
        --strategy="$s" "vdb/e1:1024|e2:1024|z:$K" \
done