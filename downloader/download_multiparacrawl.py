#!/usr/bin/env python3
"""Make the NLLB training sets"""
import datasets
import os
import tqdm
from utils import MULTI_PARA_CRAWL

def download(l1, l2, size, seed):
    pair = f"{l1}-{l2}"
    inv_pair = f"{l2}-{l1}"
    
    folder = f"data/{pair}"
    os.makedirs(folder, exist_ok=True)

    try:
        ds = datasets.load_dataset("multi_para_crawl", lang1=MULTI_PARA_CRAWL[l1], lang2=MULTI_PARA_CRAWL[l2], trust_remote_code=True)
        ds = ds["train"]
    except Exception:
        ds = datasets.load_dataset("multi_para_crawl", lang1=MULTI_PARA_CRAWL[l2], lang2=MULTI_PARA_CRAWL[l1], trust_remote_code=True)
        ds = ds["train"]

    if size == -1:
        # All data    
        with open(os.path.join(folder, f"train.{l1}"), "w") as f1, open(os.path.join(folder, f"train.{l2}"), "w") as f2:
            for row in tqdm.tqdm(ds, leave=False):
                translation = row["translation"] 
                f1.write(f"{translation[MULTI_PARA_CRAWL[l1]].replace(u"\u2029", "").replace(u"\u2028", "").strip()}\n")
                f2.write(f"{translation[MULTI_PARA_CRAWL[l2]].replace(u"\u2029", "").replace(u"\u2028", "").strip()}\n")
    else:
        small = ds.shuffle(seed=seed)
        small = small.take(size)

        with open(os.path.join(folder, f"train.{size}.{l1}"), "w") as f1, \
            open(os.path.join(folder, f"train.{size}.{l2}"), "w") as f2:
            for row in tqdm.tqdm(small, leave=False):
                translation = row["translation"] 
                f1.write(f"{translation[MULTI_PARA_CRAWL[l1]].replace(u"\u2029", "").replace(u"\u2028", "").strip()}\n")
                f2.write(f"{translation[MULTI_PARA_CRAWL[l2]].replace(u"\u2029", "").replace(u"\u2028", "").strip()}\n")

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
