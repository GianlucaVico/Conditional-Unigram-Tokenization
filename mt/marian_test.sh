#!/bin/bash
set -e -x
if [ $# -lt 11 ]; then
    echo "Usage: $0 <config> <model> <src> <trg> <src_vocab> <trg_vocab> <output_tok> <output_detok> <output_scores> <output_comet> <gpus>"
    exit 1
fi

MARIAN="path-to-marian-folder"

MARIAN_TRAIN=$MARIAN/marian
MARIAN_DECODER=$MARIAN/marian-decoder
MARIAN_VOCAB=$MARIAN/marian-vocab
MARIAN_SCORER=$MARIAN/marian-scorer


CONFIG=$1
MODEL=$2
SRC=$3
TRG=$4
SRC_VOCAB=$5
TRG_VOCAB=$6
OUTPUT_TOK=$7
OUTPUT_DETOK=$8
OUTPUT_SCORES=$9
OUTPUT_COMET=${10}
GPUS=${@:11}

OUTPUT_DIR=$(dirname $MODEL)
nvidia-smi

# Translate
$MARIAN_DECODER -m $MODEL --config $CONFIG \
    --input $SRC --output $OUTPUT_TOK \
    --vocabs $SRC_VOCAB $TRG_VOCAB --no-spm-decode \
    --mini-batch 64 --maxi-batch 100 \
    --devices $GPUS \

# Detokenize
sed -e 's/ //g ; s/▁/ /g' -e 's/^[[:blank:]]*//;s/[[:blank:]]*$//' $OUTPUT_TOK > $OUTPUT_DETOK

# Evaluate
sacrebleu $TRG --input $OUTPUT_DETOK -m bleu chrf ter -cw 2 --format json -q > $OUTPUT_SCORES # sacrebleu $1 -m bleu -b -w 4
comet-score -s $SRC -t $OUTPUT_DETOK -r $TRG --quiet --only_system --gpus 1 > $OUTPUT_COMET
