#!/usr/bin/env python3
"""Make the NLLB training sets"""
import datasets
import os
import tqdm
from utils import CODES, FLORES_SPLITS

def download(l1, l2):
    pair = f"{l1}-{l2}"
    inv_pair = f"{l2}-{l1}"
    lang1 = CODES[l1]
    lang2 = CODES[l2]
    lang_pair = f"{lang1}-{lang2}"
    inv_lang_pair = f"{lang2}-{lang1}"
    
    folder = f"data/{pair}"
    os.makedirs(folder, exist_ok=True)

    try:
        ds = datasets.load_dataset("facebook/flores", lang_pair, verification_mode=None, trust_remote_code=True)
    except Exception:
        ds = datasets.load_dataset("facebook/flores", inv_lang_pair, verification_mode=None, trust_remote_code=True)

    col1 = f"sentence_{lang1}"
    col2 = f"sentence_{lang2}"

    # All data    
    for split in FLORES_SPLITS:
        tmp = ds[split]        
        with open(os.path.join(folder, f"flores.{split}.{l1}"), "w") as f1, open(os.path.join(folder, f"flores.{split}.{l2}"), "w") as f2:            
            for row in tmp:
                f1.write(row[col1].strip() + "\n")
                f2.write(row[col2].strip() + "\n")

    if not os.path.exists(f"data/{inv_pair}"):    
        os.symlink(os.path.realpath(folder), f"data/{inv_pair}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs=2, default=(None, None))

    args = parser.parse_args()

    l1, l2 = args.langs
    download(l1, l2)