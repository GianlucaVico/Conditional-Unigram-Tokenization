#!/usr/bin/env python3
import argparse
import sys
import sentencepiece as spm
from paired_sp import prepare
import fileinput
import os
import re


def train(data: list[str], output: str, vocab: int, type: str = "unigram", do_prepare: bool = True, hard: bool = False) -> None:
    try:
        spm.SentencePieceTrainer.train(
            sentence_iterator = map(prepare, data) if do_prepare else data,
            model_prefix = output,
            vocab_size = vocab,
            model_type = type,
            character_coverage = 1.0,
            shrinking_factor = 0.75,
            num_sub_iterations = 2,
            allow_whitespace_only_pieces = True, # Default is False
            add_dummy_prefix = True,
            remove_extra_whitespaces = True,
            hard_vocab_limit = hard,
            byte_fallback = True
        )
    except RuntimeError as err:
        # Vocabulary is too large
        err_msg = str(err)
        regex_match = re.search(r"(?<=<=) \d*", err_msg)
        if regex_match is not None:
            # Fix
            old_size = vocab
            new_size = int(regex_match.group())
            print(
                f"Vocabulary size for SP spans is too high ({old_size}). Set it to {new_size}.",
                file=sys.stderr,
            )
            spm.SentencePieceTrainer.train(
                sentence_iterator = map(prepare, data) if do_prepare else data,
                model_prefix = output,
                vocab_size = new_size,
                model_type = type,
                character_coverage = 1.0,
                shrinking_factor = 0.75,
                num_sub_iterations = 2,
                allow_whitespace_only_pieces = True, # Default is False
                add_dummy_prefix = True,
                remove_extra_whitespaces = True,
                hard_vocab_limit = hard,
                byte_fallback = True
            )
        else:
            raise err

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", nargs="+", action="extend", type=str, help="Input files")
    parser.add_argument("-o", "--output", type=str, help="Output model")
    parser.add_argument("-v", "--vocab", type=int, default=16000, help="Vocab size")
    parser.add_argument("--type", choices=["unigram", "bpe"], default="unigram")
    parser.add_argument("--no-prepare", action="store_false")
    parser.add_argument("--hard", action="store_true", help="Use hard vocab limit.")
    args = parser.parse_args()

    do_prepare = args.no_prepare
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with fileinput.input(args.input) as f:
        train(f, args.output, args.vocab, args.type, do_prepare, args.hard)
