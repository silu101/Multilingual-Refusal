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
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import mmengine

# Packages imported by pipeline/ and scripts/ that are missing from this
# fork's requirements.txt (verified by grep: mmengine, deep_translator and
# jsonpickle are used in dataset/load_dataset.py, pipeline/run_pipeline.py,
# scripts/multi_test.py but absent from requirements.txt). Installing them
# is necessary for the official code to run at all; it changes no method
# code. See CHANGES.md.
MISSING_DEPS = ["mmengine==0.10.4", "deep-translator==1.11.4", "jsonpickle==3.2.2"]


def pip_install_missing():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + MISSING_DEPS, check=True)


def build_source_config(model_path: str, source_lang: str, model_alias: str) -> mmengine.Config:
    template_path = f"pipeline/runs/{model_alias}/{source_lang}/{source_lang}.yaml"
    if not os.path.exists(template_path):
        template_path = os.path.join(os.path.dirname(__file__), "en.yaml")
        assert source_lang == "en", (
            f"No checked-in config template for source_lang={source_lang!r} "
            f"and it isn't 'en' (the only language with a fallback template "
            f"in sagemaker_tier1/). Add pipeline/runs/{model_alias}/{source_lang}/"
            f"{source_lang}.yaml before running this source language."
        )
    cfg = mmengine.Config.fromfile(template_path)
    cfg.model_path = model_path
    cfg.source_lang = source_lang
    cfg.lang = source_lang
    cfg.artifact_path = (
        f"pipeline/runs/{model_alias}" if source_lang == "en"
        else f"pipeline/runs/{model_alias}/{source_lang}"
    )
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--source_lang", required=True)
    p.add_argument("--target_langs", required=True, help="comma-separated language codes")
    args = p.parse_args()

    pip_install_missing()

    # Imported after MISSING_DEPS are installed.
    from pipeline.run_pipeline import run_pipeline
    from scripts.multi_test import main as multi_test_main

    model_alias = os.path.basename(args.model_path)
    cfg = build_source_config(args.model_path, args.source_lang, model_alias)
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
