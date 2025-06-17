#!/usr/bin/env python3
import os
import argparse
import sentencepiece as spm
from paired_sp.utils import prepare

def tokenize(data: list[str], output: str, model: str, do_prepare: bool = True) -> None:
    sp = spm.SentencePieceProcessor(model_file=model)
    lines = map(prepare, data) if do_prepare else data
    with open(output, "w") as f:
        for line in lines:
            toks = sp.encode(line, out_type=str)
            f.write(" ".join(toks) + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input file")
    parser.add_argument("-o", "--output", type=str, help="Output file")
    parser.add_argument("-m", "--model", help="SP model")    
    args = parser.parse_args()

    dir_ = os.path.dirname(args.output)
    if dir_ != "":
        os.makedirs(dir_, exist_ok=True)
    with open(args.input) as f:
        tokenize(f, args.output, args.model)
