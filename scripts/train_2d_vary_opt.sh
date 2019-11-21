#!/usr/env bash

DATASET=$1

python3 train.py --epoch 100 --dataset DATASET --strategy="oneshot"  "vdb/e1:1024|e2:1024|z:2"
python3 train.py --epoch 100 --dataset DATASET --strategy="seq/d:1|e:5"  "vdb/e1:1024|e2:1024|z:2"
python3 train.py --epoch 100 --dataset DATASET --strategy="seq/d:1|e:10"  "vdb/e1:1024|e2:1024|z:2"
python3 train.py --epoch 100 --dataset DATASET --strategy="seq/d:1|e:20"  "vdb/e1:1024|e2:1024|z:2"