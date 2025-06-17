#!/bin/bash

if [ $# -ne 4 ]; then
    echo "Usage: $0 <tokenizer> <src_dir> <trg_dir> <output_dir>" >&2
    exit 1
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo Variable SLURM_ARRAY_TASK_ID is not set >&2
    exit 1
fi

if [ -z "$SLURM_ARRAY_TASK_COUNT" ]; then
    echo Variable SLURM_ARRAY_TASK_COUNT is not set >&2
    exit 1
fi

TOKENIZER=$1
INPUT_SRC_DIR=$2
INPUT_TRG_DIR=$3
OUTPUT_DIR=$4

printf -v FILE_NAME "%0${#SLURM_ARRAY_TASK_COUNT}d" $SLURM_ARRAY_TASK_ID
python -m paired_sp tokenize -m cond-exact --model-file $TOKENIZER --src $INPUT_SRC_DIR/$FILE_NAME.tok --trg $INPUT_TRG_DIR/$FILE_NAME -o $OUTPUT_DIR/$FILE_NAME.tok
 