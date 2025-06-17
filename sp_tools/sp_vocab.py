#!/usr/bin/env python3
import argparse
import yaml
import sentencepiece as spm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Marian")
    parser.add_argument("-m", "--model", type=str, required=True, help="Sentencepiece model.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output voabulary")

    args = parser.parse_args()
    model = spm.SentencePieceProcessor(args.model)

    vocab = {model.id_to_piece(i): i for i in range(model.piece_size())}

    with open(args.output, "w") as f:
        yaml.dump(vocab, f, indent=2)
