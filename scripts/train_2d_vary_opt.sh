#!/usr/bin/env bash

DATASET=$1
BETA=$2
M=$3
K=$4

time python3 train.py --epoch 100 --dataset $DATASET --beta $BETA -M $M --strategy="oneshot"  "vdb/e1:1024|e2:1024|z:$K"
time python3 train.py --epoch 100 --dataset $DATASET --beta $BETA -M $M --strategy="seq/d:1|e:5"  "vdb/e1:1024|e2:1024|z:$K"
time python3 train.py --epoch 100 --dataset $DATASET --beta $BETA -M $M --strategy="seq/d:1|e:10"  "vdb/e1:1024|e2:1024|z:$K"
time python3 train.py --epoch 100 --dataset $DATASET --beta $BETA -M $M --strategy="seq/d:1|e:20"  "vdb/e1:1024|e2:1024|z:$K"
time python3 train.py --epoch 100 --dataset $DATASET --beta $BETA -M $M --strategy="seq/d:5|e:1"  "vdb/e1:1024|e2:1024|z:$K"
time python3 train.py --epoch 100 --dataset $DATASET --beta $BETA -M $M --strategy="seq/d:10|e:1"  "vdb/e1:1024|e2:1024|z:$K"
time python3 train.py --epoch 100 --dataset $DATASET --beta $BETA -M $M --strategy="seq/d:20|e:1"  "vdb/e1:1024|e2:1024|z:$K"