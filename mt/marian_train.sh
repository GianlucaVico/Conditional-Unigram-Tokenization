#!/bin/bash
if [ $# -lt 10 ]; then
    echo "Usage: $0 <config> <model> <src> <trg> <src_valid> <trg_valid> <src_vocab> <trg_vocab> <seed> <gpus>"
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
SRC_VALID=$5
TRG_VALID=$6
SRC_VOCAB=$7
TRG_VOCAB=$8
SEED=$9
GPUS=${@:10}

OUTPUT_DIR=$(dirname $MODEL)
mkdir -p $OUTPUT_DIR
nvidia-smi

$MARIAN_TRAIN -m $MODEL --config $CONFIG --tied-embeddings \
    --train-sets $SRC $TRG --valid-sets $SRC_VALID $TRG_VALID \
    --vocabs $SRC_VOCAB $TRG_VOCAB \
    --valid-script-path mt/validate.sh \
    --valid-script-args $OUTPUT_DIR $TRG_VALID \
    --valid-translation-output $OUTPUT_DIR/validation-{E}-{U}.txt \
    --overwrite \
    --no-reload \
    --seed $SEED \
    --devices $GPUS    
