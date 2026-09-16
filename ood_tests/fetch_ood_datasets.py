#!/usr/bin/env python
"""
Fetches the semantic-content OOD datasets named in "OOD Generalization Tests
for Mechanistic LLM Safety" (Table 2) that this fork does not already ship,
and formats each into the repo's own schema: a JSON list of
{"instruction": str, "category": str | None}, matching
dataset/processed/advbench.json, harmbench_test.json, etc. exactly, so they
load through dataset/load_dataset.py's existing load_dataset() unmodified.

Already present in this fork (no fetching needed): advbench, harmbench_test,
harmbench_val, tdc2023, strongreject, malicious_instruct (note: the actual
filename is malicious_instruct.json with an underscore, even though
dataset/load_dataset.py's PROCESSED_DATASET_NAMES lists "maliciousinstruct"
without one -- a pre-existing naming inconsistency in the official repo, not
introduced here; load it via the literal filename, not
load_dataset("maliciousinstruct")).

Datasets fetched here (all public, non-gated, verified via the HF Hub API
before writing this file -- see run_ood_eval.py's module docstring for the
full list of what was checked and how):

  beavertails        PKU-Alignment/BeaverTails (round0, 30k test split)
  anthropic_redteam  gohsyi/red-team-attempts -- a public, non-gated mirror
                     of Anthropic's original red-team-attempts.jsonl.gz
                     release (Ganguli et al., 2022). NOT an official
                     Anthropic HF dataset; provenance is a third-party
                     mirror, flagged here rather than silently treated as
                     primary-source.
  openai_moderation  mmathys/openai-moderation-api-evaluation
  aegis2             nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (test split,
                     prompt_label == "unsafe" only)
  simplesafetytests  walledai/SimpleSafetyTests (the "instruct" split)

Usage:
    python -m ood_tests.fetch_ood_datasets [--out-dir dataset/processed] [--limit N]

Requires the `datasets` package (not currently in requirements.txt or
sagemaker_tier1/requirements_fixed.txt -- add it before running this on
SageMaker; see ood_tests/README.md).
"""
import argparse
import json
import os

# NOTE: import is deferred into main() rather than done at module load time,
# matching this fork's established pattern elsewhere (see
# sagemaker_tier1/entrypoint_tier1.py) for a package that may not yet be
# installed in the calling environment.


def _write(records, out_dir, name):
    path = os.path.join(out_dir, f"{name}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} items -> {path}")


def fetch_beavertails(load_dataset, limit=None):
    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    seen = set()
    out = []
    for row in ds:
        if row["is_safe"]:
            continue
        prompt = row["prompt"].strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        violated = [k for k, v in row["category"].items() if v]
        out.append({"instruction": prompt, "category": ", ".join(violated) if violated else None})
        if limit and len(out) >= limit:
            break
    return out


def fetch_anthropic_redteam(load_dataset, limit=None, min_task_harmlessness_score=0.5):
    # min_task_harmlessness_score filters to attempts the red-teamer's own
    # task_description was scored as more clearly harmful in intent (higher
    # task_descripton_harmlessness_score = more harmful), since this dataset
    # skews toward graphic offensive-language content more than the
    # actionable-technical-harm style of AdvBench/HarmBench -- see
    # ood_tests/README.md's caveat on this dataset specifically.
    ds = load_dataset("gohsyi/red-team-attempts", split="train")
    seen = set()
    out = []
    for row in ds:
        score = row.get("task_descripton_harmlessness_score")
        if score is None or score < min_task_harmlessness_score:
            continue
        prompt_turns = row.get("prompt") or []
        if not prompt_turns:
            continue
        first_human = next((t["content"] for t in prompt_turns if t.get("role") == "user"), None)
        if not first_human:
            continue
        first_human = first_human.strip()
        if not first_human or first_human in seen:
            continue
        seen.add(first_human)
        out.append({"instruction": first_human, "category": None})
        if limit and len(out) >= limit:
            break
    return out


def fetch_openai_moderation(load_dataset, limit=None):
    ds = load_dataset("mmathys/openai-moderation-api-evaluation", split="train")
    flag_cols = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]
    out = []
    for row in ds:
        flagged = [c for c in flag_cols if row.get(c)]
        if not flagged:
            continue
        prompt = row["prompt"].strip()
        if not prompt:
            continue
        out.append({"instruction": prompt, "category": ", ".join(flagged)})
        if limit and len(out) >= limit:
            break
    return out


def fetch_aegis2(load_dataset, limit=None):
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="test")
    seen = set()
    out = []
    for row in ds:
        if row.get("prompt_label") != "unsafe":
            continue
        prompt = (row.get("prompt") or "").strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        out.append({"instruction": prompt, "category": row.get("violated_categories") or None})
        if limit and len(out) >= limit:
            break
    return out


def fetch_simplesafetytests(load_dataset, limit=None):
    ds = load_dataset("walledai/SimpleSafetyTests", split="instruct")
    out = []
    for row in ds:
        prompt = row["prompt"].strip()
        if not prompt:
            continue
        out.append({"instruction": prompt, "category": row.get("harm_type")})
        if limit and len(out) >= limit:
            break
    return out


FETCHERS = {
    "beavertails": fetch_beavertails,
    "anthropic_redteam": fetch_anthropic_redteam,
    "openai_moderation": fetch_openai_moderation,
    "aegis2": fetch_aegis2,
    "simplesafetytests": fetch_simplesafetytests,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "dataset", "processed"))
    p.add_argument("--limit", type=int, default=None, help="Cap each dataset to this many items (for a quick smoke test).")
    p.add_argument("--only", nargs="*", choices=list(FETCHERS.keys()), default=None, help="Fetch only these datasets.")
    args = p.parse_args()

    from datasets import load_dataset

    names = args.only or list(FETCHERS.keys())
    for name in names:
        print(f"=== Fetching {name} ===")
        records = FETCHERS[name](load_dataset, limit=args.limit)
        _write(records, args.out_dir, name)


if __name__ == "__main__":
    main()
