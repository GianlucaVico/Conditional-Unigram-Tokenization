#!/bin/bash
#SBATCH -J lm-fra-ita
#SBATCH -o lm-fra-ita.out
#SBATCH -e lm-fra-ita.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G

# Run fra-ita manually
# flores is already tokenized
# the train test is already split in 20 partitions, ~7M lines each.
# We need 2M examples

set -e

SAMPLES=2000000
TMP_DIR=`mktemp -d -p .`

echo "Created TMP_DIR $TMP_DIR"

function cleanup {   
    if [[ "$TMP_DIR" && -d "$TMP_DIR" ]]; then
        rm -rd "$TMP_DIR"
        echo "Deleted temp working directory $TMP_DIR"
    fi    
}

# trap cleanup EXIT


if [[ ! "$TMP_DIR" || ! -d "$TMP_DIR" ]]; then
    echo "Could not create temp dir"
    exit 1
fi

head -n $SAMPLES data/fra-ita/train.fra_parts/00 > $TMP_DIR/train.fra
head -n $SAMPLES data/fra-ita/train.ita_parts/00 > $TMP_DIR/train.ita
echo "Copied training data"

# TOKENIZE SP
for VOCAB in 8000 16000 32000 ; do
    python sp_tools/sp_tokenize.py -i $TMP_DIR/train.fra -m models/ita-fra/$VOCAB/1000000/unigram.fra.model -o $TMP_DIR/unigram.$VOCAB.fra
    python sp_tools/sp_tokenize.py -i $TMP_DIR/train.ita -m models/fra-ita/$VOCAB/1000000/unigram.ita.model -o $TMP_DIR/unigram.$VOCAB.ita
done
echo "Started SP tokenization"


# SP LM
for VOCAB in 8000 16000 32000 ; do
    for SEED in 3 5 42 ; do
        sbatch -J b-fra-$VOCAB-$SEED -o $TMP_DIR/b-fra-$VOCAB-$SEED.out -e $TMP_DIR/b-fra-$VOCAB-$SEED.err \
            -n 1 -N 1 -c 1 --gpus=1 --mem=25G \
            ./lm_sp_train.sh $TMP_DIR $VOCAB $SEED
        sbatch -J b-ita-$VOCAB-$SEED -o $TMP_DIR/b-ita-$VOCAB-$SEED.out -e $TMP_DIR/b-ita-$VOCAB-$SEED.err \
            -n 1 -N 1 -c 1 --gpus=1 --mem=25G \
            ./lm_sp_train.sh $TMP_DIR $VOCAB $SEED "other"
    done
done

echo "Started baseline training"

# TOKENIZE USP
mkdir $TMP_DIR/fra_parts
mkdir $TMP_DIR/ita_parts
split -a 2 -d -n r/20 $TMP_DIR/train.fra $TMP_DIR/fra_parts/
split -a 2 -d -n r/20 $TMP_DIR/train.ita $TMP_DIR/ita_parts/
echo "Split data"

for VOCAB in 8000 16000 32000 ; do # TODO restart from here
    mkdir $TMP_DIR/fra_parts_$VOCAB
    sbatch --wait -J fra-$VOCAB -o $TMP_DIR/fra-$VOCAB.out -a 0-19 -c 1 \
        --mem=4000 ./mt/tokenize_usp.sh models/ita-fra/$VOCAB/1000000/align/fra.tar.gz \
            $TMP_DIR/fra_parts/ $TMP_DIR/fra_parts_$VOCAB
   
    cat $TMP_DIR/fra_parts_$VOCAB/* > $TMP_DIR/psp.$VOCAB.fra
    mkdir $TMP_DIR/ita_parts_$VOCAB
    sbatch --wait -J ita-$VOCAB -o $TMP_DIR/ita-$VOCAB.out -a 0-19 -c 1 \
        --mem=4000 ./mt/tokenize_usp.sh models/fra-ita/$VOCAB/1000000/align/ita.tar.gz \
            $TMP_DIR/ita_parts/ $TMP_DIR/ita_parts_$VOCAB
    cat $TMP_DIR/ita_parts_$VOCAB/* > $TMP_DIR/psp.$VOCAB.ita
done
echo "UPS tokenization"

# PSP LM
for VOCAB in 8000 16000 32000 ; do
    for SEED in 3 5 42 ; do
        sbatch -J fra-$VOCAB-$SEED -o $TMP_DIR/fra-$VOCAB-$SEED.out -e $TMP_DIR/fra-$VOCAB-$SEED.err \
            -n 1 -N 1 -c 1 --gpus=1 --mem=25G \
            ./lm_psp_train.sh $TMP_DIR $VOCAB $SEED
        sbatch -J ita-$VOCAB-$SEED -o $TMP_DIR/ita-$VOCAB-$SEED.out -e $TMP_DIR/ita-$VOCAB-$SEED.err \
            -n 1 -N 1 -c 1 --gpus=1 --mem=25G \
            ./lm_psp_train.sh $TMP_DIR $VOCAB $SEED "other"
    done
done

echo "Started LM training"
