#!/usr/bin/env python
"""
Semantic-content OOD evaluation for the refusal-direction ablation method
(Arditi et al. / Wang et al.), following "OOD Generalization Tests for
Mechanistic LLM Safety" (Table 2)'s design: extract the refusal direction on
one in-distribution dataset (AdvBench), then measure how much the ablation
effect (baseline compliance -> post-ablation compliance) holds up on
datasets the direction never saw.

This calls pipeline.run_pipeline.run_pipeline() and
pipeline.submodules.evaluate_jailbreak.evaluate_jailbreak() UNMODIFIED (same
principle as sagemaker_tier1/entrypoint_tier1.py for the Tier-1 replication)
-- the only repo-code changes needed were: dataset/load_dataset.py's
HARMTYPES/PROCESSED_DATASET_NAMES additions (see that file's comments), and
nothing else. Everything OOD-specific lives in this ood_tests/ directory.

Design, mapped to Table 2 of the OOD paper:
  1. Extract + select the refusal direction using PURE AdvBench train/val
     (ood_tests/split_advbench.py's output, harmtype='harmful_advbench') --
     NOT the official repo's default 'harmful' harmtype, which mixes
     AdvBench + MaliciousInstruct + TDC2023 and would contaminate any later
     OOD test against MaliciousInstruct specifically.
  2. Baseline eval on HarmBench's test set, DEDUPED against AdvBench first
     (ood_tests/dedup.py) -- the OOD paper's own explicit caution ("overlaps
     with AdvBench, so deduplicate before cross-dataset OOD"), confirmed
     necessary in this exact repo (dataset/splits/harmful_val.json's one
     sample is textually identical to harmbench_val.json's).
  3. OOD eval on datasets the direction never saw: BeaverTails, the
     Anthropic red-team-attempts mirror, OpenAI Moderation, Aegis 2.0, and
     SimpleSafetyTests (fetched by ood_tests/fetch_ood_datasets.py).
  4. Report baseline vs. ablated compliance for every dataset, so the
     ablation-effect *lift* on each OOD set can be compared against the
     HarmBench (in-family) baseline lift.

Not in scope here (see ood_tests/README.md): attack/jailbreak-family OOD
(Table 3 -- GCG/AutoDAN/PAIR) and the full utility suite (Table 4 -- only
substring-matching/WildGuard compliance is measured here, not
MMLU/GSM8K/BBH/TruthfulQA/XSTest).

Prerequisites (see ood_tests/README.md for exact commands):
  python -m ood_tests.split_advbench
  python -m ood_tests.fetch_ood_datasets
(run_ood_eval.py's main() also runs the HarmBench dedup step itself, on the
fly, if dataset/processed/harmbench_test_deduped.json doesn't exist yet.)

Usage:
    python -m ood_tests.run_ood_eval --model_path google/gemma-2b-it
"""
import argparse
import json
import os
import random
import shutil
import tempfile


REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")

DEFAULT_OOD_DATASETS = [
    "harmbench_test_deduped",
    "beavertails",
    "anthropic_redteam",
    "openai_moderation",
    "aegis2",
    "simplesafetytests",
]


def ensure_harmbench_deduped():
    out_path = os.path.join(REPO_ROOT, "dataset", "processed", "harmbench_test_deduped.json")
    if os.path.exists(out_path):
        return
    from ood_tests.dedup import dedup

    with open(os.path.join(REPO_ROOT, "dataset", "processed", "harmbench_test.json")) as f:
        harmbench = json.load(f)
    with open(os.path.join(REPO_ROOT, "dataset", "processed", "advbench.json")) as f:
        advbench = json.load(f)
    kept, dropped = dedup(harmbench, advbench, threshold=0.85)
    print(f"HarmBench dedup: {len(harmbench)} total, {len(dropped)} dropped as AdvBench near-duplicates, {len(kept)} kept")
    with open(out_path, "w") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)


def build_config(model_path, model_alias):
    import mmengine

    # Reuses sagemaker_tier1/en.yaml as a generic hyperparameter template --
    # every field in it besides the ones overridden below is
    # language/dataset-agnostic (n_train, kl thresholds, batch_size, etc.);
    # see that file and entrypoint_tier1.py's build_source_config() for the
    # same pattern applied to the Tier-1 replication.
    template_path = os.path.join(REPO_ROOT, "sagemaker_tier1", "en.yaml")
    cfg = mmengine.Config.fromfile(template_path)
    cfg.model_path = model_path
    cfg.source_lang = "en"
    cfg.lang = "en"
    cfg.harmtype_2 = "harmful_advbench"  # pure AdvBench, not the mixed default 'harmful'
    cfg.artifact_path = os.path.join("ood_tests", "runs", model_alias)
    # The MMLU/wikitext/truthfulqa capability check via lm_eval is a
    # separate concern from this experiment's actual target (compliance
    # lift on OOD datasets) and needs a separate install this fork's
    # requirements don't provide -- see run_pipeline.py's own
    # skip_eval_harness guard, added during the Tier-1 replication.
    cfg.skip_eval_harness = True
    return cfg


def evaluate_dataset(model_base, direction, cfg, dataset_name, n_sample, batch_size, max_new_tokens=512):
    from dataset.load_dataset import load_dataset
    from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
    from pipeline.utils.hook_utils import get_all_direction_ablation_hooks

    data = load_dataset(dataset_name)
    if n_sample is not None and len(data) > n_sample:
        random.seed(cfg.random_seed)
        data = random.sample(data, n_sample)

    baseline_completions = model_base.generate_completions(
        data, fwd_pre_hooks=[], fwd_hooks=[], max_new_tokens=max_new_tokens, batch_size=batch_size, system=None
    )
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction, cfg.start_layer, cfg.ablation_coeff)
    ablated_completions = model_base.generate_completions(
        data, fwd_pre_hooks=ablation_fwd_pre_hooks, fwd_hooks=ablation_fwd_hooks, max_new_tokens=max_new_tokens, batch_size=batch_size, system=None
    )

    out_dir = os.path.join(cfg.artifact_path, "ood_completions")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset_name}_baseline_completions.json"), "w") as f:
        json.dump(baseline_completions, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, f"{dataset_name}_ablated_completions.json"), "w") as f:
        json.dump(ablated_completions, f, indent=2, ensure_ascii=False)

    baseline_eval = evaluate_jailbreak(
        completions=baseline_completions,
        methodologies=cfg.jailbreak_eval_methodologies,
        evaluation_path=os.path.join(out_dir, f"{dataset_name}_baseline_evaluations.json"),
        cfg=cfg,
        logger=None,
    )
    ablated_eval = evaluate_jailbreak(
        completions=ablated_completions,
        methodologies=cfg.jailbreak_eval_methodologies,
        evaluation_path=os.path.join(out_dir, f"{dataset_name}_ablated_evaluations.json"),
        cfg=cfg,
        logger=None,
    )

    baseline_compliance = 1 - baseline_eval.get("wildguard_refusal", float("nan"))
    ablated_compliance = 1 - ablated_eval.get("wildguard_refusal", float("nan"))
    return {
        "n": len(data),
        "baseline_compliance": baseline_compliance,
        "ablated_compliance": ablated_compliance,
        "lift": ablated_compliance - baseline_compliance,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="google/gemma-2b-it")
    p.add_argument("--ood_datasets", nargs="+", default=DEFAULT_OOD_DATASETS)
    p.add_argument("--n_ood_sample", type=int, default=None, help="Subsample each OOD dataset to this many prompts (fixed seed). Unset = use every prompt in the dataset.")
    p.add_argument("--wildguard_batch_size", type=int, default=8)
    p.add_argument("--generation_batch_size", type=int, default=None, help="Overrides cfg.batch_size for model generation. Unset = use the template's default (64).")
    args = p.parse_args()

    ensure_harmbench_deduped()

    import mmengine  # noqa: F401 -- imported here only to fail fast with a clear error if missing
    from pipeline.run_pipeline import run_pipeline
    from pipeline.model_utils.model_factory import construct_model_base

    model_alias = os.path.basename(args.model_path)
    cfg = build_config(args.model_path, model_alias)
    cfg.wildguard_batch_size = args.wildguard_batch_size
    if args.generation_batch_size:
        cfg.batch_size = args.generation_batch_size
    tmp_cfg_path = tempfile.mktemp(suffix=".yaml")
    cfg.dump(tmp_cfg_path)

    print("=== [1/2] Extracting refusal direction from pure AdvBench ===", flush=True)
    run_pipeline(config_path=tmp_cfg_path, model_path=args.model_path, batch_size=None)

    # Bridge the direction_ablation.pt -> direction.pt naming mismatch
    # between pipeline/run_pipeline.py's save name and what a fresh load
    # here expects -- same pre-existing repo inconsistency documented in
    # sagemaker_tier1/CHANGES.md for the Tier-1 replication.
    saved = os.path.join(cfg.artifact_path, "direction_ablation.pt")
    assert os.path.exists(saved), f"run_pipeline did not produce {saved}"

    print("=== [2/2] Evaluating baseline vs. ablated compliance on each OOD dataset ===", flush=True)
    import torch

    direction = torch.load(saved)
    if isinstance(direction, list):
        direction = direction[0]

    model_base = construct_model_base(args.model_path)

    results = {}
    for name in args.ood_datasets:
        print(f"--- {name} ---", flush=True)
        results[name] = evaluate_dataset(
            model_base, direction, cfg, name, args.n_ood_sample, cfg.batch_size
        )
        r = results[name]
        print(f"    n={r['n']}  baseline={r['baseline_compliance']:.3f}  ablated={r['ablated_compliance']:.3f}  lift={r['lift']:+.3f}")
        # Same fix applied to sagemaker_tier1/entrypoint_tier1.py's
        # per-target loop, for the same reason: CUDA allocator
        # fragmentation across many sequential generate() calls within one
        # long-running process, confirmed there via a small (480MiB)
        # allocation failure that smaller batch sizes alone didn't fix.
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    results_path = os.path.join(cfg.artifact_path, "ood_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {results_path}")

    harmbench_lift = results.get("harmbench_test_deduped", {}).get("lift")
    if harmbench_lift is not None:
        print("\nLift relative to the HarmBench (in-family) baseline:")
        for name, r in results.items():
            if name == "harmbench_test_deduped":
                continue
            print(f"  {name}: {r['lift']:+.3f} (HarmBench: {harmbench_lift:+.3f}, ratio={r['lift']/harmbench_lift if harmbench_lift else float('nan'):.2f})")


if __name__ == "__main__":
    main()
