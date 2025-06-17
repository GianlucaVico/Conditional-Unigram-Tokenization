#!/usr/bin/env python3
"""Make the NLLB training sets"""
import datasets
import os
import tqdm
from utils import CODES

def download(l1, l2, size, seed):
    pair = f"{l1}-{l2}"
    inv_pair = f"{l2}-{l1}"
    lang1 = CODES[l1]
    lang2 = CODES[l2]
    lang_pair = f"{lang1}-{lang2}"
    inv_lang_pair = f"{lang2}-{lang1}"
    
    folder = f"data/{pair}"
    os.makedirs(folder, exist_ok=True)

    try:
        ds = datasets.load_dataset(
            "allenai/nllb", 
            lang_pair, 
            verification_mode=None, 
            trust_remote_code=True, 
            split="train", 
            streaming=True
        )
    except Exception:
        ds = datasets.load_dataset(
            "allenai/nllb", 
            inv_lang_pair, 
            verification_mode=None, 
            trust_remote_code=True, 
            split="train", 
            streaming=True
        )

    if size == -1:
        # All data    
        with open(os.path.join(folder, f"train.{l1}"), "w") as f1, open(os.path.join(folder, f"train.{l2}"), "w") as f2:
            for sample in tqdm.tqdm(ds, leave=False):
                f1.write(sample["translation"][lang1].replace(u"\u2029", "").replace(u"\u2028", "").strip() + "\n")
                f2.write(sample["translation"][lang2].replace(u"\u2029", "").replace(u"\u2028", "").strip() + "\n")
    else:
        small = ds.shuffle(buffer_size=size, seed=seed)
        small = small.take(size)

        with open(os.path.join(folder, f"train.{size}.{l1}"), "w") as f1, \
            open(os.path.join(folder, f"train.{size}.{l2}"), "w") as f2:
            for sample in tqdm.tqdm(small, leave=False, total=size):
                f1.write(sample["translation"][lang1].replace(u"\u2029", " ").replace(u"\u2028", " ").strip() + "\n")
                f2.write(sample["translation"][lang2].replace(u"\u2029", " ").replace(u"\u2028", " ").strip() + "\n")

    if not os.path.exists(f"data/{inv_pair}"):    
        os.symlink(os.path.realpath(folder), f"data/{inv_pair}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs=2, default=(None, None))
    parser.add_argument("--size", default=-1, type=int)
    parser.add_argument("--seed", type=int)  

    args = parser.parse_args()

    l1, l2 = args.langs
    seed = None if args.seed is None else int(args.seed)
    download(l1, l2, int(args.size), seed)
