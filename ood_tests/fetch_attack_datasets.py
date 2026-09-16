#!/usr/bin/env python
"""
Fetches the two attack/jailbreak-family OOD resources from "OOD
Generalization Tests for Mechanistic LLM Safety" (Table 3) that are
actually *datasets* -- as opposed to GCG, AutoDAN, and PAIR, which are live
attack *algorithms* that would need to run against the target model with
their own (heavy) codebases. See ood_tests/README.md's "Attack-family OOD"
section for why those three are out of scope here.

  jailbreakllms_wrapped   Held-out AdvBench goals wrapped inside in-the-wild
                          DAN/persona/roleplay templates (Shen et al., 2024,
                          "Do Anything Now" -- TrustAIRLab/in-the-wild-
                          jailbreak-prompts). The paper's own framing (§3):
                          "The harmful goal and its jailbreak wrapper must
                          be stored separately" -- this is exactly that:
                          the SAME goal set used for extraction, now
                          wrapped in a template family the direction never
                          saw, holding the attack transformation itself out
                          rather than the semantic content.
  wildjailbreak_adversarial  Pre-composed adversarial harmful prompts from
                          allenai/wildjailbreak's `eval` config
                          (data_type == "adversarial_harmful"). Ships
                          already-wrapped, not goal+template pairs, so this
                          one is a different (not goal-controlled) style of
                          attack-family holdout -- a different wrapper
                          distribution, not the same goals under a new
                          wrapper. Gated (self-serve, "auto") on the HF
                          Hub; requires the same kind of one-time access
                          acceptance as gemma-2b-it/allenai/wildguard did
                          during the Tier-1 replication.

Template scope note: TrustAIRLab/in-the-wild-jailbreak-prompts has 1405
templates total, but only 94 contain an explicit "[INSERT PROMPT HERE]"
insertion point that a goal can be substituted into as a single-turn
prompt. The other ~1311 are "persona-priming" templates meant to be sent as
a first conversational turn, with the actual harmful ask following in a
second turn -- using those would need multi-turn generation support this
repo's pipeline doesn't have (model_base.generate_completions() is
single-turn only). Restricted to the 94 with an explicit placeholder,
disclosed here rather than silently narrowed.

Usage:
    python -m ood_tests.fetch_attack_datasets [--out-dir dataset/processed] [--n-wrapped 100] [--limit N]
"""
import argparse
import json
import os
import random

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
PLACEHOLDER = "[INSERT PROMPT HERE]"


def _write(records, out_dir, name):
    path = os.path.join(out_dir, f"{name}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} items -> {path}")


def fetch_jailbreakllms_wrapped(load_dataset, n_wrapped=100, seed=1, goals_path=None, max_template_chars=1500):
    """Pairs held-out AdvBench goals 1:1 with a sample of placeholder
    templates (not a full cross product, to keep the eval set a comparable
    size to the other OOD sets).

    max_template_chars filters out unusually long templates before pairing.
    Real distribution (checked against live data): of the 94 templates with
    an explicit placeholder, lengths range 260-6499 chars (median ~1812,
    mean ~2106). Confirmed necessary via a real failure: even at
    generation_batch_size=2, gemma's prefill logit computation
    (proportional to prompt_length x vocab_size, and Gemma's vocab is
    ~256k) OOM'd on job mr-ood-attack-2026-09-16-11-11-34-844 on this
    dataset specifically -- a single long template can exceed available
    memory regardless of how small the batch is, so this needed to be a
    length filter, not another batch-size reduction. 1500 chars keeps 37 of
    94 templates (verified against live data), comfortably enough for a
    n_wrapped=50 sample."""
    ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", split="train")
    all_templates = [row["prompt"] for row in ds if PLACEHOLDER in row["prompt"]]
    templates = [t for t in all_templates if len(t) <= max_template_chars]
    print(f"  {len(all_templates)} of {len(ds)} in-the-wild templates have an explicit {PLACEHOLDER!r} insertion point; "
          f"{len(templates)} of those are <= {max_template_chars} chars")

    goals_path = goals_path or os.path.join(REPO_ROOT, "dataset", "processed", "advbench.json")
    with open(goals_path) as f:
        goals = [d["instruction"] for d in json.load(f)]

    random.seed(seed)
    n = min(n_wrapped, len(templates), len(goals))
    sampled_templates = random.sample(templates, n)
    sampled_goals = random.sample(goals, n)

    out = []
    for template, goal in zip(sampled_templates, sampled_goals):
        wrapped = template.replace(PLACEHOLDER, goal)
        out.append({"instruction": wrapped, "category": None})
    return out


def fetch_wildjailbreak_adversarial(load_dataset, limit=None):
    ds = load_dataset("allenai/wildjailbreak", "eval", split="train")
    out = []
    for row in ds:
        if row.get("data_type") != "adversarial_harmful":
            continue
        text = (row.get("adversarial") or "").strip()
        if not text:
            continue
        out.append({"instruction": text, "category": "adversarial_harmful"})
        if limit and len(out) >= limit:
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "dataset", "processed"))
    p.add_argument("--n-wrapped", type=int, default=100, help="How many (goal, template) pairs to produce for jailbreakllms_wrapped.")
    p.add_argument("--limit", type=int, default=None, help="Cap wildjailbreak_adversarial to this many items.")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    from datasets import load_dataset

    print("=== Fetching jailbreakllms_wrapped ===")
    wrapped = fetch_jailbreakllms_wrapped(load_dataset, n_wrapped=args.n_wrapped, seed=args.seed)
    _write(wrapped, args.out_dir, "jailbreakllms_wrapped")

    print("=== Fetching wildjailbreak_adversarial ===")
    wj = fetch_wildjailbreak_adversarial(load_dataset, limit=args.limit)
    _write(wj, args.out_dir, "wildjailbreak_adversarial")


if __name__ == "__main__":
    main()
