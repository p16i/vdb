STRATEGIES=$2
DATASET=$1
K=256
M=1
OUTPUT="./artifacts/$DATASET--bottleneck-curve"

BETA="0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001"

for s in $STRATEGIES; do
    for b in $BETA; do
        echo "$DATASET $s $b"
        time python3 train.py --epoch 100 --dataset $DATASET  \
            --beta $b -M $M --strategy=$s  "vdb/e1:1024|e2:1024|z:$K" \
            --output-dir $OUTPUT
    done
done