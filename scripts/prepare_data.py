"""Tokenize a text dataset into a flat uint16 token file for pretraining.

Example:
    python scripts/prepare_data.py --dataset openwebtext --out data/
"""

import argparse
import os

import numpy as np
from tqdm import tqdm

from minigpt.data.tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="openwebtext", help="HF datasets name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/")
    ap.add_argument("--val-frac", type=float, default=0.0005)
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer()

    ds = load_dataset(args.dataset, split=args.split)
    split = ds.train_test_split(test_size=args.val_frac, seed=42)

    for name, dset in [("train", split["train"]), ("val", split["test"])]:
        # First pass: count tokens so we can preallocate the memmap.
        lengths = []
        for ex in tqdm(dset, desc=f"counting {name}"):
            lengths.append(len(tok.encode(ex["text"], add_eot=True)))
        total = int(np.sum(lengths))

        path = os.path.join(args.out, f"{name}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
        offset = 0
        for ex in tqdm(dset, desc=f"writing {name}"):
            ids = tok.encode(ex["text"], add_eot=True)
            arr[offset: offset + len(ids)] = ids
            offset += len(ids)
        arr.flush()
        print(f"wrote {total:,} tokens -> {path}")


if __name__ == "__main__":
    main()
