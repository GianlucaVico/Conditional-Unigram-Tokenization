#!/bin/bash

TMP_DIR=$1
VOCAB=$2
SEED=$3
DIRECTION=$4 


# empty: one direction, not empty: other direction
if [ -z $DIRECTION ] ; then
    pair="fra-ita"
    trg="ita"
    src="fra"
else
    pair="ita-fra"
    trg="fra"
    src="ita"
fi

valid="data/$pair/psp/$VOCAB/1000000/align/flores.dev.$trg.tok.unpaired"
test="data/$pair/psp/$VOCAB/1000000/align/flores.devtest.$trg.tok.unpaired"
vocab="models/$pair/$VOCAB/1000000/align/$trg.yml"
vocab2="models/$trg-$src/$VOCAB/1000000/unigram.$src.yml"
output_dir="models/lm_bi/$pair/$VOCAB/1000000/align/seed_$SEED"

mkdir -p $output_dir

/usr/bin/time -v ./lm/train.py \
    --train $TMP_DIR/psp.$VOCAB.$trg \
    --train-other $TMP_DIR/unigram.$VOCAB.$src \
    --valid $valid \
    --test $test \
    --vocab $vocab \
    --additional-vocab $vocab2\
    --output-dir $output_dir \
    --steps 2000000 \
    --batch 64 \
    --seed $SEED
rm -rd $output_dir/checkpoint-*
