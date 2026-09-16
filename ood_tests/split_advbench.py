#!/usr/bin/env python
"""
Splits dataset/processed/advbench.json (520 items, shipped as one flat list,
no train/val split) into dataset/splits/harmful_advbench_train.json and
harmful_advbench_val.json, matching the file-naming convention
dataset/load_dataset.py's load_dataset_split() already expects
(SPLIT_DATASET_FILENAME = "splits/{harmtype}_{split}.json").

Why this exists: the official repo's default harmful_train.json/
harmful_val.json (used by pipeline.run_pipeline unless overridden) are a MIX
of AdvBench + MaliciousInstruct + TDC2023 (confirmed directly: harmful_val's
one sample here is textually identical to harmbench_val's -- these splits
already overlap with data this OOD experiment needs to hold out). Extracting
the refusal direction on that mix would contaminate any later "OOD" test
against MaliciousInstruct specifically, and the paper's own semantic-OOD
design (see ood_tests/README.md) requires a *pure* AdvBench-only train/val
set instead.

Sizes: 520 total advbench.json items, split 80/20 with a fixed seed for
reproducibility -- 416 train / 104 val. (The official harmful_train/val use
128/32; this uses a larger val set since AdvBench alone, unlike the official
mixed pool, has no other source to draw from, and a bigger held-out val set
gives select_direction's KL-gated sweep more signal.)

Usage:
    python -m ood_tests.split_advbench [--train-size 416] [--seed 1]
"""
import argparse
import json
import os
import random

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--advbench-path", default=os.path.join(REPO_ROOT, "dataset", "processed", "advbench.json"))
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "dataset", "splits"))
    p.add_argument("--train-size", type=int, default=416)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    with open(args.advbench_path) as f:
        items = json.load(f)

    random.seed(args.seed)
    shuffled = items[:]
    random.shuffle(shuffled)

    train = shuffled[: args.train_size]
    val = shuffled[args.train_size :]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "harmful_advbench_train.json")
    val_path = os.path.join(args.out_dir, "harmful_advbench_val.json")
    with open(train_path, "w") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(val_path, "w") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    print(f"advbench.json: {len(items)} total -> {len(train)} train / {len(val)} val")
    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")


if __name__ == "__main__":
    main()
