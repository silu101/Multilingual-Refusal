#!/usr/bin/env python
"""
SageMaker training entry point for the Tier-1 replication
(English/de/zh/th refusal-direction extraction + 14-language cross-lingual
ablation eval, on Qwen2.5-7B-Instruct).

This file is deploy/orchestration glue only. It does not change any method
code in pipeline/ or scripts/ -- it calls pipeline.run_pipeline.run_pipeline()
and scripts.multi_test.main() exactly as the README's own CLI usage does,
just looped over target languages and driven by CLI args instead of a
hand-edited YAML per run. See CHANGES.md in this directory for the full
list of what was added to this fork and why.

For --source_lang <L>:
  1. Loads the config template for L. de/ja/ko/ru/th/yo/zh already have one
     checked in at pipeline/runs/<model_alias>/<L>/<L>.yaml (from the
     upstream authors' own runs). 'en' has no template in this fork, so we
     ship sagemaker_tier1/en.yaml, which mirrors that same schema exactly
     (only lang/source_lang/artifact_path differ) -- see CHANGES.md.
  2. Calls pipeline.run_pipeline.run_pipeline() unmodified to extract and
     KL-filter-select the refusal direction.
  3. WORKAROUND for a path-naming mismatch between the two official
     scripts: pipeline/run_pipeline.py's select_and_save_direction() saves
     the chosen vector as "<artifact_path>/direction_ablation.pt", but
     scripts/multi_test.py loads "<artifact_path>/direction.pt" (no
     "_ablation" suffix). Neither script renames the file, so multi_test.py
     cannot find what run_pipeline.py just wrote. We copy
     direction_ablation.pt -> direction.pt after step 2, which is exactly
     what a human running these two scripts back-to-back by hand would
     have had to do. No vector data is altered by this copy.
  4. For each --target_langs entry, builds an in-memory copy of the same
     config with cfg.lang set to that target, and calls
     scripts.multi_test.main() unmodified on it.
  5. Copies pipeline/runs/ and output/ into $SM_MODEL_DIR so results reach
     S3 as the training job's model artifact.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

# NOTE: mmengine is intentionally NOT imported here at module level -- it is
# one of the packages pip_install_missing() installs at runtime (see below),
# so it isn't present yet when this file is first loaded. It's imported
# locally, after pip_install_missing() has run, inside main() and
# build_source_config(). `from __future__ import annotations` above keeps
# the `-> mmengine.Config` type hint from being evaluated at module-load
# time (it'd otherwise NameError before the local import ever runs).

# requirements.txt in this fork is NOT installed on the SageMaker container
# as-is. Verified reason (not a guess): SageMaker's PyTorch estimator
# auto-runs `pip install -r requirements.txt` from source_dir before the
# entry point, and this fails outright -- litellm==1.40.9 has been yanked
# from PyPI and is no longer installable by anyone, on any machine, as of
# this run. That is independent of instance type or CUDA version (confirmed:
# the same "ERROR: No matching distribution found for litellm==1.40.9" is
# pip's own resolver failing before any package is even downloaded).
#
# Fix: sagemaker_tier1/requirements_fixed.txt is a full copy of
# requirements.txt with exactly two kinds of edits: (1) the `litellm==1.40.9`
# line changed to unpinned `litellm` (nothing in pipeline/ or scripts/
# actually calls litellm's API unless the "llamaguard2" jailbreak_eval
# methodology is selected, which no config in this fork uses -- it's only
# imported at module level in pipeline/submodules/evaluate_jailbreak.py and
# must resolve for that import to succeed, nothing more), and (2) three
# packages appended that requirements.txt never listed at all despite being
# imported by dataset/load_dataset.py, pipeline/run_pipeline.py and
# scripts/multi_test.py: mmengine, deep-translator, jsonpickle. Every other
# pin is identical to the original file (diff the two to confirm). An
# earlier version of this function hand-picked a subset of packages instead
# of using the full corrected file, which missed several imports actually
# needed by pipeline/run_pipeline.py's dependency chain (e.g. `datasets`)
# and had to be fixed via repeated trial-and-error on real SageMaker jobs --
# using the full file avoids that class of bug entirely. launch_tier1.py
# excludes the original requirements.txt from the staged upload so
# SageMaker's broken auto-install step is skipped; this function installs
# from requirements_fixed.txt instead. See CHANGES.md.
REQUIREMENTS_FIXED = os.path.join(os.path.dirname(__file__), "requirements_fixed.txt")


def pip_install_missing(max_attempts: int = 3):
    # Retries a transient mid-download connection failure (observed once:
    # an SSL socket read reset partway through a large wheel download,
    # unrelated to any package pin) rather than requiring a manual relaunch.
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", REQUIREMENTS_FIXED], check=True)
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            print(f"pip install attempt {attempt}/{max_attempts} failed, retrying: {e}", flush=True)
    raise last_err


def build_source_config(model_path: str, source_lang: str, model_alias: str) -> "mmengine.Config":
    import mmengine  # see module-level note: installed at runtime, imported locally
    # Prefer a model-and-language-specific template if one is already
    # checked in (de/ja/ko/ru/th/yo/zh have one for Qwen2.5-7B-Instruct from
    # the upstream authors' own runs); otherwise fall back to
    # sagemaker_tier1/en.yaml as a generic hyperparameter template for ANY
    # source language -- every field in it besides lang/source_lang/
    # artifact_path/model_path is language-agnostic (n_train, kl thresholds,
    # batch_size, etc.), and all four of those fields are overwritten below
    # regardless of which template was loaded. (Originally this fallback was
    # asserted to only apply to source_lang=="en"; relaxed to any language
    # once we needed a template for a model -- gemma-2b-it -- that has no
    # checked-in per-language configs at all yet.)
    template_path = f"pipeline/runs/{model_alias}/{source_lang}/{source_lang}.yaml"
    if not os.path.exists(template_path):
        template_path = os.path.join(os.path.dirname(__file__), "en.yaml")
    cfg = mmengine.Config.fromfile(template_path)
    cfg.model_path = model_path
    cfg.source_lang = source_lang
    cfg.lang = source_lang
    cfg.artifact_path = (
        f"pipeline/runs/{model_alias}" if source_lang == "en"
        else f"pipeline/runs/{model_alias}/{source_lang}"
    )
    # run_pipeline() unconditionally runs an MMLU/wikitext/truthfulqa/
    # arc_challenge capability check via lm_eval at the end, regardless of
    # this cfg's other settings -- see pipeline/run_pipeline.py's own
    # skip_eval_harness guard for the full rationale. Not part of Tier-1's
    # scope (Figures 1 & 2) and requires a separate install this fork's
    # requirements.txt doesn't provide.
    cfg.skip_eval_harness = True
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="google/gemma-2b-it")
    p.add_argument("--source_lang", required=True)
    p.add_argument("--target_langs", required=True, help="comma-separated language codes")
    # Cost-control knobs (see scripts/multi_test.py's guarded opt-in and
    # CHANGES.md): default None/False here too, so running this file
    # directly with no flags matches official full-scale behavior.
    p.add_argument("--test_sample_size", type=int, default=None,
                    help="If set, subsample each language's 572-prompt test set to this many (fixed seed=cfg.random_seed). Unset = full 572, official behavior.")
    # NOTE: SageMaker's PyTorch estimator always invokes the entry point as
    # `--key value` for every hyperparameter (it doesn't support bare flags),
    # so this takes an explicit string rather than being a store_true flag,
    # which would break under that invocation style.
    p.add_argument("--skip_addition", type=str, default="false",
                    help="'true' to skip generating/evaluating the activation-addition completions (Figure 3, not part of Tier-1's ablation-only scope). Default 'false': generate them, official behavior.")
    p.add_argument("--wildguard_batch_size", type=int, default=1,
                    help="Completions per WildGuard generate() call (see evaluators/wildguard.py). Default 1 = original unbatched behavior.")
    args = p.parse_args()
    args.skip_addition = args.skip_addition.lower() in ("true", "1", "yes")

    pip_install_missing()

    # Imported after pip_install_missing() has run -- mmengine included.
    import mmengine
    from pipeline.run_pipeline import run_pipeline
    from scripts.multi_test import main as multi_test_main

    model_alias = os.path.basename(args.model_path)
    cfg = build_source_config(args.model_path, args.source_lang, model_alias)
    cfg.wildguard_batch_size = args.wildguard_batch_size
    tmp_cfg_path = tempfile.mktemp(suffix=".yaml")
    cfg.dump(tmp_cfg_path)

    print(f"=== [1/2] pipeline.run_pipeline: extracting direction for source_lang={args.source_lang} ===", flush=True)
    run_pipeline(config_path=tmp_cfg_path, model_path=args.model_path, batch_size=None)

    # Bridge the direction_ablation.pt -> direction.pt naming mismatch
    # (see module docstring, point 3).
    saved = os.path.join(cfg.artifact_path, "direction_ablation.pt")
    expected_by_multi_test = os.path.join(cfg.artifact_path, "direction.pt")
    assert os.path.exists(saved), f"run_pipeline did not produce {saved}"
    shutil.copy(saved, expected_by_multi_test)
    print(f"Copied {saved} -> {expected_by_multi_test} (multi_test.py's expected filename)", flush=True)

    targets = [t.strip() for t in args.target_langs.split(",") if t.strip()]
    failures = []
    for i, target in enumerate(targets):
        print(f"=== [2/2] scripts.multi_test {i + 1}/{len(targets)}: source={args.source_lang} target={target} ===", flush=True)
        eval_cfg = mmengine.Config.fromfile(tmp_cfg_path)
        eval_cfg.lang = target
        eval_cfg.source_lang = args.source_lang
        eval_cfg.artifact_path = f"output/tier1_crosslingual/{model_alias}/{args.source_lang}/{target}"
        if args.test_sample_size is not None:
            eval_cfg.test_sample_size = args.test_sample_size
        if args.skip_addition:
            eval_cfg.skip_addition = True
        eval_tmp = tempfile.mktemp(suffix=".yaml")
        eval_cfg.dump(eval_tmp)
        try:
            multi_test_main(eval_tmp)
        except Exception:
            print(f"!!! FAILED source={args.source_lang} target={target}, continuing with remaining targets !!!", flush=True)
            traceback.print_exc()
            failures.append(target)

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    for src, dst in [("pipeline/runs", "pipeline_runs"), ("output", "output")]:
        if os.path.exists(src):
            shutil.copytree(src, os.path.join(model_dir, dst), dirs_exist_ok=True)

    if failures:
        print(f"Completed with failures on targets: {failures}", flush=True)
        sys.exit(1)
    print("All targets completed successfully.", flush=True)


if __name__ == "__main__":
    main()
