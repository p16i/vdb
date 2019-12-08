#!/usr/bin/env bash

DATASET=$1

ALGOSLUG=`echo $SCRIPT | sed s/[.]/-/g | sed s/py$//g`
OUTPUT="./artifacts/$ALGOSLUG/cherry-pick-after-merging-$DATASET--lr-decay-various-opts-M$M"

#STRATEGIES="oneshot seq/d:5|e:1 seq/d:10|e:1 seq/d:20|e:1"
#STRATEGIES="seq/d:5|e:1 seq/d:10|e:1 seq/d:20|e:1"
#STRATEGIES="seq/d:10|e:1"
STRATEGIES="oneshot"
#STRATEGIES="alt/e:5|d:1 alt/e:10|d:1 alt/e:20|d:1"

for i in `seq $TOTAL_RUNS`; do
        echo "------ RUN $i ------"
        for s in $STRATEGIES; do
                echo "Script: $SCRIPT |  STATS:$s DATASET=$DATASET BETA=$BETA M=$M K=$K -> $OUTPUT"
            sbatch ./slurm_train.sh \
                $SCRIPT \
                --epoch $EPOCH \
                --dataset $DATASET --beta $BETA -M $M \
                --output-dir="$OUTPUT" \
                --strategy="$s" "vdb/e1:1024|e2:1024|z:$K"
           sleep 3 # preventing naming cache
        done
done