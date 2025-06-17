#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 <tokenizer> <input_dir> <output_dir>" >&2
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
INPUT_DIR=$2
OUTPUT_DIR=$3

printf -v FILE_NAME "%0${#SLURM_ARRAY_TASK_COUNT}d" $SLURM_ARRAY_TASK_ID

python sp_tools/sp_tokenize.py -i $INPUT_DIR/$FILE_NAME -o $OUTPUT_DIR/$FILE_NAME.tok -m $TOKENIZER
