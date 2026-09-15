#!/usr/bin/env python
"""
Launch Tier-1 cross-lingual refusal-direction replication jobs on SageMaker
(Qwen2.5-7B-Instruct only, per user decision -- gemma-2-9b-it deferred).

One job per --source_lang, each running entrypoint_tier1.py, which extracts
the direction for that source language and then evaluates it against every
language in --target_langs (default: all 14 PolyRefuse languages). Jobs are
launched with wait=False so multiple source languages run in parallel.

Reuses the IAM role, region and PyTorch-estimator conventions already
established in ~/Downloads/safety-layers-repro/sagemaker/ (a *different*,
unrelated paper-reproduction repo -- see CHANGES.md for why this project
lives in its own folder instead of inside that one). Only the HF token file
is shared (it is a generic Hugging Face token, not specific to that repo);
the Anthropic key from that repo is intentionally NOT used here, since this
paper's own method uses WildGuard as the compliance judge, not Claude.

Usage:
    python sagemaker_tier1/launch_tier1.py --source_langs en,de,zh,th \
        --target_langs ar,de,en,es,fr,it,ja,ko,nl,pl,ru,th,yo,zh \
        --max_run_hours 8
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# This repo's working tree carries .git/ (128M) and a checked-in output/
# (581M -- the upstream authors' own prior results) that our job neither
# needs to upload nor to read; SageMaker's source_dir tars whatever
# directory you point it at, so we stage a pruned copy rather than pointing
# it at the working tree directly. Nothing under output/ or .git/ is
# modified or deleted by this -- only the staged rsync copy excludes them.
#
# requirements.txt is ALSO excluded: SageMaker's PyTorch estimator
# auto-installs source_dir/requirements.txt before the entry point runs,
# and this fork's requirements.txt cannot install at all right now --
# litellm==1.40.9 has been yanked from PyPI (confirmed via a real failed
# job: "ERROR: No matching distribution found for litellm==1.40.9"), so
# `pip install -r requirements.txt` fails on any machine today, not just
# SageMaker. entrypoint_tier1.py installs a working dependency set itself
# instead (same pins throughout except litellm, left unpinned). The
# original requirements.txt file itself is untouched in the repo. See
# CHANGES.md.
EXCLUDE_FROM_UPLOAD = [".git", "output", "__pycache__", ".pytest_cache", "requirements.txt"]

ROLE_ARN = "arn:aws:iam::344977996863:role/safety-layers-sagemaker-execution-role"
REGION = "us-east-1"
DEFAULT_INSTANCE_TYPE = "ml.g6e.xlarge"  # 48GB L40S, single GPU -- our job needs
    # Qwen2.5-7B-Instruct and WildGuard (~7B) resident on GPU simultaneously
    # (the official evaluate_jailbreak() code doesn't unload the target model
    # before loading WildGuard), ~28-30GB combined in bf16. A single 24GB GPU
    # (g5.xlarge/2xlarge/4xlarge/8xlarge/16xlarge) will likely OOM. g5.12xlarge
    # (4x A10G, 96GB total) would also work: pipeline/model_utils/qwen2_model.py
    # and evaluators/wildguard.py both already load with device_map="auto", so
    # accelerate will shard each model across the 4 GPUs automatically -- no
    # code change needed for that path either.
DEFAULT_MODEL_PATH = "google/gemma-2b-it"  # switched from Qwen2.5-7B-Instruct by
    # user decision, to fit single-24GB-GPU instances (g5.xlarge, confirmed
    # live capacity) instead of requiring the scarcer 48GB/multi-GPU tiers.
    # gemma-2b-it is one of the paper's own benchmarked models (Fig. 1), so
    # this is a different valid data point from the paper, not a deviation.

# 'yo' (Yoruba) excluded by user decision: Amazon Translate returns
# UnsupportedLanguagePairException for yo->en (confirmed locally), and
# switching translation providers was necessary because Google Translate's
# free endpoint was persistently blocked from this AWS account (see
# sagemaker_tier1/CHANGES.md). This is a real coverage gap against the
# paper's 14-language claim, not a silent omission -- Yoruba is the
# paper's headline "safety-misaligned language" finding (Table 1).
ALL_LANGS = "ar,de,en,es,fr,it,ja,ko,nl,pl,ru,th,zh"


def hf_token() -> str:
    f = Path.home() / ".hf_token_safety_layers"
    if not f.exists():
        raise SystemExit(f"No HF token found at {f}.")
    return f.read_text().strip()


def stage_source_dir(repo_root: Path) -> str:
    staging = Path(tempfile.mkdtemp(prefix="mr_tier1_src_"))
    rsync_excludes = []
    for e in EXCLUDE_FROM_UPLOAD:
        rsync_excludes += ["--exclude", e]
    subprocess.run(
        ["rsync", "-a", *rsync_excludes, f"{repo_root}/", f"{staging}/"],
        check=True,
    )
    return str(staging)


def launch(source_lang: str, target_langs: str, max_run_hours: float, session: sagemaker.Session, source_dir: str, instance_type: str, model_path: str, test_sample_size: int | None, skip_addition: bool, wildguard_batch_size: int):
    hyperparameters = {
        "model_path": model_path,
        "source_lang": source_lang,
        "target_langs": target_langs,
        "wildguard_batch_size": wildguard_batch_size,
    }
    if test_sample_size is not None:
        hyperparameters["test_sample_size"] = test_sample_size
    if skip_addition:
        hyperparameters["skip_addition"] = "true"
    estimator = PyTorch(
        entry_point="sagemaker_tier1/entrypoint_tier1.py",
        source_dir=source_dir,
        role=ROLE_ARN,
        framework_version="2.3",
        py_version="py311",
        instance_type=instance_type,
        instance_count=1,
        volume_size=100,
        sagemaker_session=session,
        base_job_name=f"mr-tier1-{source_lang}",
        max_run=int(max_run_hours * 60 * 60),
        environment={
            "HF_TOKEN": hf_token(),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        hyperparameters=hyperparameters,
    )
    estimator.fit(wait=False)
    print(f"[{source_lang}] submitted: {estimator.latest_training_job.name} (model={model_path})")
    return estimator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_langs", default="en,de,zh,th")
    p.add_argument("--target_langs", default=ALL_LANGS)
    p.add_argument("--max_run_hours", type=float, default=8.0)
    p.add_argument("--instance_type", default=DEFAULT_INSTANCE_TYPE)
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--test_sample_size", type=int, default=None,
                    help="Subsample each language's 572-prompt test set to this many (fixed seed). Unset = full 572.")
    p.add_argument("--skip_addition", action="store_true",
                    help="Skip the activation-addition completions/eval (Figure 3, out of Tier-1 scope) to cut ~1/3 of the compute.")
    p.add_argument("--wildguard_batch_size", type=int, default=1,
                    help="Completions per WildGuard generate() call. Default 1 = original unbatched behavior (measured ~12.7s/item; batching is the dominant lever for the 5-GPU-hour budget).")
    args = p.parse_args()

    session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
    repo_root = Path(__file__).resolve().parent.parent
    source_dir = stage_source_dir(repo_root)
    print(f"Staged pruned source_dir at {source_dir} (excludes {EXCLUDE_FROM_UPLOAD})")

    jobs = []
    for source_lang in args.source_langs.split(","):
        jobs.append(launch(source_lang.strip(), args.target_langs, args.max_run_hours, session, source_dir, args.instance_type, args.model_path, args.test_sample_size, args.skip_addition, args.wildguard_batch_size))

    print("\nAll jobs submitted (wait=False). Job names:")
    for e in jobs:
        print(" ", e.latest_training_job.name)


if __name__ == "__main__":
    main()
