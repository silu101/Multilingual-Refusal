#!/usr/bin/env python
"""
Removes near-duplicate prompts between two instruction lists, matching the
schema used throughout dataset/processed/*.json:
[{"instruction": str, "category": str | None}, ...].

Why this exists: "OOD Generalization Tests for Mechanistic LLM Safety"
(Table 2) explicitly warns that HarmBench "overlaps with AdvBench, so
deduplicate before cross-dataset OOD." Confirmed directly in this repo:
dataset/splits/harmful_val.json's one sample is textually identical to
dataset/processed/harmbench_val.json's. Exact-match dedup alone would miss
paraphrased overlaps (same underlying behavior, reworded), so this uses a
normalized-text similarity threshold instead of pure string equality.

Method: lowercase + whitespace-normalize each instruction, then compare
every candidate against every reference instruction with
difflib.SequenceMatcher's ratio() (stdlib, no new dependency). A candidate
is dropped if its best match against any reference exceeds --threshold.
This is a real but modest bar (O(n*m) string comparisons) -- fine at the
scale used here (HarmBench's ~200 items against AdvBench's ~520), not
intended for much larger corpora.
"""
import argparse
import difflib
import json
import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def dedup(candidates, references, threshold=0.85):
    """Returns (kept, dropped) -- both lists of the original candidate dicts."""
    ref_normalized = [normalize(r["instruction"]) for r in references]
    kept, dropped = [], []
    for item in candidates:
        cand_norm = normalize(item["instruction"])
        best = max(
            (difflib.SequenceMatcher(None, cand_norm, r).ratio() for r in ref_normalized),
            default=0.0,
        )
        if best >= threshold:
            dropped.append(item)
        else:
            kept.append(item)
    return kept, dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True, help="Path to the dataset to filter (e.g. dataset/processed/harmbench_test.json)")
    p.add_argument("--references", required=True, nargs="+", help="Path(s) to dataset(s) to check overlap against (e.g. dataset/processed/advbench.json)")
    p.add_argument("--out", required=True, help="Path to write the deduplicated result")
    p.add_argument("--threshold", type=float, default=0.85)
    args = p.parse_args()

    with open(args.candidates) as f:
        candidates = json.load(f)

    references = []
    for ref_path in args.references:
        with open(ref_path) as f:
            references.extend(json.load(f))

    kept, dropped = dedup(candidates, references, threshold=args.threshold)

    with open(args.out, "w") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print(f"{args.candidates}: {len(candidates)} total, {len(dropped)} dropped as near-duplicates of {args.references}, {len(kept)} kept -> {args.out}")
    if dropped:
        print("Example dropped instruction(s):")
        for d in dropped[:3]:
            print(f"  - {d['instruction'][:120]}")


if __name__ == "__main__":
    main()
