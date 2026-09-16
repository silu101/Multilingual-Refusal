#!/usr/bin/env python
"""
SageMaker training entry point for ood_tests/run_ood_eval.py. Deploy glue
only, same principle as sagemaker_tier1/entrypoint_tier1.py: installs
dependencies, runs the already-tested ood_tests/*.py scripts as subprocesses
(not imported directly, since they each parse sys.argv themselves via
argparse -- calling them as `python -m ood_tests.X` avoids argv conflicts
with this entry point's own arguments), then copies results into
$SM_MODEL_DIR.

Usage (via ood_tests/launch_ood.py, not normally called directly):
    python sagemaker_entrypoint.py --suite semantic --model_path google/gemma-2b-it \
        --n_ood_sample 100 --wildguard_batch_size 8
"""
import argparse
import os
import shutil
import subprocess
import sys

REQUIREMENTS_FIXED = os.path.join(os.path.dirname(__file__), "..", "sagemaker_tier1", "requirements_fixed.txt")

SEMANTIC_DATASETS = ["harmbench_test_deduped", "beavertails", "anthropic_redteam", "openai_moderation", "aegis2", "simplesafetytests"]
ATTACK_DATASETS = ["harmbench_test_deduped", "jailbreakllms_wrapped", "wildjailbreak_adversarial"]


def run(cmd, max_attempts=1):
    print("+", " ".join(cmd), flush=True)
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            if attempt < max_attempts:
                print(f"attempt {attempt}/{max_attempts} failed, retrying: {e}", flush=True)
    raise last_err


def pip_install_missing():
    # Same requirements_fixed.txt as the Tier-1 replication (see
    # sagemaker_tier1/CHANGES.md for why requirements.txt itself can't
    # install), plus `datasets` for the fetch scripts, which nothing in
    # the Tier-1 path needed.
    run([sys.executable, "-m", "pip", "install", "-q", "-r", REQUIREMENTS_FIXED], max_attempts=3)
    run([sys.executable, "-m", "pip", "install", "-q", "datasets"], max_attempts=3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["semantic", "attack"], required=True)
    p.add_argument("--model_path", default="google/gemma-2b-it")
    p.add_argument("--n_ood_sample", type=int, default=100)
    p.add_argument("--wildguard_batch_size", type=int, default=8)
    p.add_argument("--fetch_limit", type=int, default=None, help="Cap each fetched dataset to this many items (smoke-test scope).")
    p.add_argument("--generation_batch_size", type=int, default=None,
                    help="Overrides cfg.batch_size for model generation. OOD prompts (esp. jailbreakllms_wrapped, ~500-word DAN templates) are much longer and more variable than Tier-1's curated PolyRefuse set -- confirmed via a real OOM (job mr-ood-semantic-2026-09-16-09-53-21-664, gemma's MLP forward pass) at the template's default batch_size=64. Unset = that default.")
    args = p.parse_args()

    pip_install_missing()

    run([sys.executable, "-m", "ood_tests.split_advbench"])

    if args.suite == "semantic":
        fetch_cmd = [sys.executable, "-m", "ood_tests.fetch_ood_datasets"]
        if args.fetch_limit:
            fetch_cmd += ["--limit", str(args.fetch_limit)]
        run(fetch_cmd)
        datasets = SEMANTIC_DATASETS
    else:
        fetch_cmd = [sys.executable, "-m", "ood_tests.fetch_attack_datasets"]
        if args.fetch_limit:
            fetch_cmd += ["--n-wrapped", str(args.fetch_limit), "--limit", str(args.fetch_limit)]
        run(fetch_cmd)
        datasets = ATTACK_DATASETS

    eval_cmd = [
        sys.executable, "-m", "ood_tests.run_ood_eval",
        "--model_path", args.model_path,
        "--ood_datasets", *datasets,
        "--n_ood_sample", str(args.n_ood_sample),
        "--wildguard_batch_size", str(args.wildguard_batch_size),
    ]
    if args.generation_batch_size:
        eval_cmd += ["--generation_batch_size", str(args.generation_batch_size)]
    run(eval_cmd)

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    for src, dst in [("ood_tests/runs", "ood_runs"), ("pipeline/runs", "pipeline_runs")]:
        if os.path.exists(src):
            shutil.copytree(src, os.path.join(model_dir, dst), dirs_exist_ok=True)


if __name__ == "__main__":
    main()
