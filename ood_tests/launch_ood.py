#!/usr/bin/env python
"""
Launch an OOD evaluation job (semantic-content or attack-family) on
SageMaker. Same IAM role/region/estimator conventions as
sagemaker_tier1/launch_tier1.py -- see that file's docstring for why.

This has never been run before; there is no measured GPU-hour or cost
number for it the way Tier-1 has after its own round of fixes. Start with
a small --n_ood_sample / --fetch_limit as a smoke test before a full run.

Usage:
    python ood_tests/launch_ood.py --suite semantic --instance_type ml.g5.2xlarge \
        --n_ood_sample 100 --max_run_hours 2
    python ood_tests/launch_ood.py --suite attack --instance_type ml.g5.4xlarge \
        --n_ood_sample 100 --max_run_hours 2
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

ROLE_ARN = "arn:aws:iam::344977996863:role/safety-layers-sagemaker-execution-role"
REGION = "us-east-1"
DEFAULT_MODEL_PATH = "google/gemma-2b-it"

# Same reasoning as sagemaker_tier1/launch_tier1.py's EXCLUDE_FROM_UPLOAD:
# .git and the checked-in output/ aren't needed by the job; requirements.txt
# is excluded because it can't install at all right now (see
# sagemaker_tier1/CHANGES.md) -- sagemaker_entrypoint.py installs
# requirements_fixed.txt itself instead.
EXCLUDE_FROM_UPLOAD = [".git", "output", "__pycache__", ".pytest_cache", "requirements.txt"]


def hf_token() -> str:
    f = Path.home() / ".hf_token_safety_layers"
    if not f.exists():
        raise SystemExit(f"No HF token found at {f}.")
    return f.read_text().strip()


def stage_source_dir(repo_root: Path) -> str:
    staging = Path(tempfile.mkdtemp(prefix="ood_src_"))
    rsync_excludes = []
    for e in EXCLUDE_FROM_UPLOAD:
        rsync_excludes += ["--exclude", e]
    subprocess.run(["rsync", "-a", *rsync_excludes, f"{repo_root}/", f"{staging}/"], check=True)
    return str(staging)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["semantic", "attack"], required=True)
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--instance_type", required=True, help="Use a different g5.* size per concurrent job -- each instance type has its own separate SageMaker quota slot (confirmed 1.0 each for g5.xlarge/2xlarge/4xlarge on this account).")
    p.add_argument("--n_ood_sample", type=int, default=100)
    p.add_argument("--wildguard_batch_size", type=int, default=8)
    p.add_argument("--fetch_limit", type=int, default=None, help="Cap each fetched dataset (smoke-test scope).")
    p.add_argument("--max_run_hours", type=float, default=2.0)
    args = p.parse_args()

    session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))
    repo_root = Path(__file__).resolve().parent.parent
    source_dir = stage_source_dir(repo_root)
    print(f"Staged pruned source_dir at {source_dir} (excludes {EXCLUDE_FROM_UPLOAD})")

    hyperparameters = {
        "suite": args.suite,
        "model_path": args.model_path,
        "n_ood_sample": args.n_ood_sample,
        "wildguard_batch_size": args.wildguard_batch_size,
    }
    if args.fetch_limit:
        hyperparameters["fetch_limit"] = args.fetch_limit

    estimator = PyTorch(
        entry_point="ood_tests/sagemaker_entrypoint.py",
        source_dir=source_dir,
        role=ROLE_ARN,
        framework_version="2.3",
        py_version="py311",
        instance_type=args.instance_type,
        instance_count=1,
        volume_size=100,
        sagemaker_session=session,
        base_job_name=f"mr-ood-{args.suite}",
        max_run=int(args.max_run_hours * 60 * 60),
        environment={
            "HF_TOKEN": hf_token(),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        hyperparameters=hyperparameters,
    )
    estimator.fit(wait=False)
    print(f"[{args.suite}] submitted: {estimator.latest_training_job.name} (model={args.model_path}, instance={args.instance_type})")


if __name__ == "__main__":
    main()
