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
EXCLUDE_FROM_UPLOAD = [".git", "output", "__pycache__", ".pytest_cache"]

ROLE_ARN = "arn:aws:iam::344977996863:role/safety-layers-sagemaker-execution-role"
REGION = "us-east-1"
INSTANCE_TYPE = "ml.g6e.xlarge"  # 48GB L40S -- confirmed-available instance family
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

ALL_LANGS = "ar,de,en,es,fr,it,ja,ko,nl,pl,ru,th,yo,zh"


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


def launch(source_lang: str, target_langs: str, max_run_hours: float, session: sagemaker.Session, source_dir: str):
    estimator = PyTorch(
        entry_point="sagemaker_tier1/entrypoint_tier1.py",
        source_dir=source_dir,
        role=ROLE_ARN,
        framework_version="2.3",
        py_version="py311",
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        volume_size=100,
        sagemaker_session=session,
        base_job_name=f"mr-tier1-{source_lang}",
        max_run=int(max_run_hours * 60 * 60),
        environment={
            "HF_TOKEN": hf_token(),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        hyperparameters={
            "model_path": MODEL_PATH,
            "source_lang": source_lang,
            "target_langs": target_langs,
        },
    )
    estimator.fit(wait=False)
    print(f"[{source_lang}] submitted: {estimator.latest_training_job.name}")
    return estimator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_langs", default="en,de,zh,th")
    p.add_argument("--target_langs", default=ALL_LANGS)
    p.add_argument("--max_run_hours", type=float, default=8.0)
    args = p.parse_args()

    session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
    repo_root = Path(__file__).resolve().parent.parent
    source_dir = stage_source_dir(repo_root)
    print(f"Staged pruned source_dir at {source_dir} (excludes {EXCLUDE_FROM_UPLOAD})")

    jobs = []
    for source_lang in args.source_langs.split(","):
        jobs.append(launch(source_lang.strip(), args.target_langs, args.max_run_hours, session, source_dir))

    print("\nAll jobs submitted (wait=False). Job names:")
    for e in jobs:
        print(" ", e.latest_training_job.name)


if __name__ == "__main__":
    main()
