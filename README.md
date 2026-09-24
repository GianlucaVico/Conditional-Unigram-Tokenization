# Conditional Unigram Tokenization with Parallel Data

This is the repository for the paper Conditional Unigram Tokenization with Parallel Data.

## Usage

### Training the tokeniser

Example:

`python -m paired_sp train -m cond-exact --src tokenised-source-data --trg target-data -p -v 32000 -o tokeniser.tar.gz -i 2 --sub-iter 2`

Parameters:

```
usage: Paired SentencePiece train [-h] [--no-preload] [--tqdm] [--debug] [--info] --model {max,sum,max-unpaired,sum-unpaired,cond-exact,pmi,cpmi} [--src SRC] --trg TRG [--maxlen MAXLEN] [--pretokenize]
                                  [--vocab-size VOCAB_SIZE] --output OUTPUT [--iter ITER] [--sub-iter SUB_ITER] [--reduce REDUCE] [--seed SEED] [--char-coverage CHAR_COVERAGE] [--temp TEMP]
                                  [--threshold THRESHOLD] [--spans SPANS] [--trainer {mc,em}]

options:
  -h, --help            show this help message and exit
  --no-preload          Preload in memory all the datasets.
  --tqdm                Show tqdm progress bars.
  --debug               Logger debug
  --info                Logger info
  --model {max,sum,max-unpaired,sum-unpaired,cond-exact,pmi,cpmi}, -m {max,sum,max-unpaired,sum-unpaired,cond-exact,pmi,cpmi}
                        Type of model.
  --src SRC             TOKENIZED sentences in the source language. For Unpaired models: use the same file for --trg.
  --trg TRG             PLAIN TEXT sentences in the target language.
  --maxlen MAXLEN, -l MAXLEN
                        Maximum length of the spans in the target language. -1 for unlimited length.
  --pretokenize, -p     Split the target sentences on white spaces.
  --vocab-size VOCAB_SIZE, -v VOCAB_SIZE
                        Vocabulary size, default 8000.
  --output OUTPUT, -o OUTPUT
                        Where to save the model
  --iter ITER, -i ITER  Maximum number of EM iterations. Stop early if the desired vocabulary size is reached.
  --sub-iter SUB_ITER   Reduce the vocabulary after this many iterations of the EM algorithm.
  --reduce REDUCE, -r REDUCE
                        Reduce this portion of vocabulary at each iteration.
  --seed SEED, -s SEED  RNG seed.
  --char-coverage CHAR_COVERAGE
                        Keep the fraction of most frequent character
  --temp TEMP           Temp. file for intermediate tokenization.
  --threshold THRESHOLD
                        Remove spans that occour less than the threshold.
  --spans SPANS         Use a fixed number of spans.
  --trainer {mc,em}     Trainer to use.
```


### Tokenising a text file:

Example:

`python -m paired_sp tokenize -m cond-exact --model-file tokeniser.tar.gz --src tokenised-source-data --trg target-data -o tokenised-target-data`

Parameters:

```
usage: Paired SentencePiece tokenize [-h] [--no-preload] [--tqdm] [--debug] [--info] --model {max,sum,max-unpaired,sum-unpaired,cond-exact,pmi,cpmi} --model-file MODEL_FILE [--src SRC] --trg TRG
                                     [--alignment ALIGNMENT] [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --no-preload          Preload in memory all the datasets.
  --tqdm                Show tqdm progress bars.
  --debug               Logger debug
  --info                Logger info
  --model {max,sum,max-unpaired,sum-unpaired,cond-exact,pmi,cpmi}, -m {max,sum,max-unpaired,sum-unpaired,cond-exact,pmi,cpmi}
                        Type of model.
  --model-file MODEL_FILE
                        tar.gz file that contains the model.
  --src SRC             TOKENIZED sentences in the source language. For Unpaired models: use the same file for --trg.
  --trg TRG             PLAIN TEXT sentences in the target language.
  --alignment ALIGNMENT
                        Eflomal alignment.
  --output OUTPUT, -o OUTPUT
                        Where to save the tokenized data. Default STDOUT.
```

### Exporing the vocabulary

Example:

`python -m paired_sp vocab -m cond-exact --model-file tokeniser.tar.gz -o vocab.yaml`

Parameters:

```
usage: Paired SentencePiece vocab [-h] [--no-preload] [--tqdm] [--debug] [--info] --model {max,sum,max-unpaired,sum-unpaired,cond-exact,pmi,cpmi} --model-file MODEL_FILE -o OUTPUT

options:
  -h, --help            show this help message and exit
  --no-preload          Preload in memory all the datasets.
  --tqdm                Show tqdm progress bars.
  --debug               Logger debug
  --info                Logger info
  --model {max,sum,max-unpaired,sum-unpaired,cond-exact,pmi,cpmi}, -m {max,sum,max-unpaired,sum-unpaired,cond-exact,pmi,cpmi}
                        Type of model.
  --model-file MODEL_FILE
                        tar.gz file that contains the model.
  -o OUTPUT, --output OUTPUT
                        Output file (yaml).
```

## Project Structure

### Main folder

* `Snakefile`: main snakemake file to train and score most of the models
* `Snakefile-run.sh`: example on how to run Snakefile on a Slurm cluster
* `do_lm_train.sh`, `lm_psp_train.sh`, `lm_sp_train.sh`: scripts to train some of the lm without snakemeke

### paired_sp

Main python module to training and using the PairedSP tokeniser.

### data

This folder contains the training data for the tokeniser and the tokenised data for the other tasks.
It has a subforlder for each language pair (e.g., `ces-ukr`, and the link `ukr-ces`).

The training data is in the following files (`ces-ukr` as an example):

* `flores.dev.ces`
* `flores.dev.ukr`
* `flores.devtest.ces`
* `flores.devtest.ukr`
* `train.100000.ces` (similar patter for Ukrainian and the other training sizes)
* `train.ces`
* `train.ukr`

For each language pair there is the folder `align` with the word aligned training data (e.g., `alig/train.500000.ces`) and intermediate eflomal files.

The tokenised data uses the following patterns:

* `psp/8000/100000/align/flores.dev.ces.tok`: tokenised with PairedSP trained with 100k sentences and 8k as vocabulary size
* `unigram/16000/500000/flores.devtest.ukr.tok`: tokenised with SentencePiece trained with 500k sentences and 16k as vocabulary size
* `unigram/16000/500000/align/train.500000.ces.tok`: tokenised word aligned training data

Snakemake should be able to download and tokenise the data except for `deu-hsb` with has to be downloaded manually.

### models

This folder contains the tokenisers, MT models, and LMs.

For the tokenisers, there is a subfolder for each language pair.
The patter is as follows: 

* `8000/100000/align_unigram.ukr.scores.eflomal`: alignment scores for Ukrainian SentencePiece (100k training sentences, 8k vocabs)
* `8000/100000/unigram.ukr.scores`: intrinsic score for Czech SentencePiece
* `8000/100000/unigram.ces.model`: Czech SentencePiece model
* `8000/100000/unigram.ces.vocab`: Czech SentencePiece vocabulary (generated by SentencePiece)
* `8000/100000/unigram.ukr.yml`: Ukrainian SentencePiece vocabulary (converted to paired_sp)
* `8000/100000/align/align_unigram.ukr.scores.eflomal`: PairedSP alignment scores for Ukrainian
* `8000/100000/align/ukr.scores`: PairedSP intrinsic scores
* `8000/100000/align/ukr.tar.gz`: PairedSP model
* `8000/100000/align/ukr.yml`: PairedSP vocabulary

A PairedSP model is saved in a tar.gz file with the following structure:

* `self_data.json`: the settings needed to run the tokeniser. Example:

	```json
	{
	  "vocab_size": 8000,
	  "maxlen": 23,
	  "pretokenize": true,
	  "seed": 0,
	  "type": "PairedSPExactCondModel"
	}
	```
	
* `table.tar.gz`: tar.gz file with the co-occurences table and the indexes (json files).
	* `_count`: dictionary of dictionary with the (expected) counts. The first key is the target language token id, the second key is the source language token id
	* `_span_index`: dictionary with target tokens (key) and ids (value)
	* `_token_index`: dictionary with source tokens (key) and ids (value)
In the repository there is an example of the tokeniser.


The LMs and MT models are in the folders `lm` and `mt`, which contains sub-folders for each language pair.
For the MT task the structure is the following:

* `8000/1000000/align/seed_3/model.npz`: ces->ukr model (tokeniser with 8k vocabulary size, 1M samples) with seed 3 and trained with PairedSP
* `8000/1000000/align/seed_3/model.npz.yml`: model settings (generated by MarianMT, used during training)
* `8000/1000000/baseline/seed_3/model.npz.decoder.yml`: decoder settings of the baseline model (generate by MarianMT, used for inference)
* `8000/1000000/baseline/seed_3/model.npz.comet`: baseline COMET scores
* `8000/1000000/align/seed_3/model.npz.scores`: SacreBLEU scores
* `8000/1000000/align/seed_3/model.npz.tok`: tokenised preditions on the test set
* `8000/1000000/align/seed_3/model.npz.detok`: detokenised predicitons on the test set

The structure is the same for LMs, the main differences are the following files:

* `all_results.json`: metrics (validation and testing)
* `model.safetensors`: model weights
* `test_examples.txt`: prediction examples


### downloader

Script to generate the datasets (except for `deu-hsb`).

### intrinsic

Python module to compute the intrinsic (see `python -m intrinsic --help`) and alignemt scores (see `python -m intrinsic.eflomal --help`).

### mt

Script for training the MT models.

### lm

Training script for the LM task. It uses some scripts from the `mt` folder.

### pretokenize

Script to compute the word aligned  (and phrase aligned) datasets.

### results

Plot the results, partially generate the paper's tables.

### Snakefiles

Snakemake rules to generate intermediate files.

### sp_tools

Scripts for training and using SencencePiece.

## Cite

```bibtex
@inproceedings{
	vico2025conditional,
	title={Conditional Unigram Tokenization with Parallel Data},
	author={Gianluca Vico and Jind{\v{r}}ich Libovick{\'y}},
	booktitle={Tokenization Workshop},
	year={2025},
	url={https://openreview.net/forum?id=lnWJWNA8YW}
}
```
