# Changes made for the SageMaker Tier-1 replication

Everything in this folder is **new**, added to run the official, unmodified
`pipeline/` and `scripts/` code on Amazon SageMaker. Nothing outside this
folder was changed except the one addition noted below. This file exists
because the person requesting this replication asked that any deviation
from the official repo be recorded rather than assumed.

## Scope (current)

Tier-1 replication (English/German/Chinese/Thai refusal-direction
extraction, cross-lingual ablation across all 14 PolyRefuse languages).
Budget ceiling: $100 total AWS spend.

**Model: `google/gemma-2b-it`**, switched from the originally-planned
`Qwen/Qwen2.5-7B-Instruct` after real infrastructure testing (see "Model
switch" below). gemma-2b-it is one of the paper's own benchmarked models
(Figure 1), so this is a different valid data point from the paper, not a
methodology deviation.

## Files added

- `en.yaml` — a generic run-config template (originally written for
  `source_lang=en` on Qwen2.5-7B-Instruct specifically; now used as the
  fallback template for any source language on any model that doesn't
  already have a checked-in per-language config — see
  `entrypoint_tier1.py`'s `build_source_config()`). Every hyperparameter in
  it (n_train=128, n_val=32, kl_threshold=0.1, ablate_kl_threshold=0.2,
  batch_size=64, max_new_tokens=512, random_seed=1, top_n=1, etc.) is
  copied verbatim from the checked-in `pipeline/runs/Qwen2.5-7B-Instruct/
  th/th.yaml` — no hyperparameter was changed, only `lang`/`source_lang`/
  `model_path`/`artifact_path`, and those four fields are overwritten
  programmatically for every run regardless of which template was loaded.
- `entrypoint_tier1.py` — SageMaker training entry point. Calls
  `pipeline.run_pipeline.run_pipeline()` and `scripts.multi_test.main()`
  **unmodified**, looped over target languages. See its module docstring
  for the full step-by-step. Three things worth flagging:
  1. **`requirements.txt` cannot install at all, on any machine, right
     now**: `litellm==1.40.9` has been yanked from PyPI. Confirmed via a
     real failed SageMaker job (`mr-tier1-en-2026-09-14-20-48-18-332`):
     `ERROR: No matching distribution found for litellm==1.40.9` — pip's
     resolver fails before downloading anything, independent of instance
     type or CUDA version. `launch_tier1.py` excludes `requirements.txt`
     from the staged upload so SageMaker's automatic
     `pip install -r requirements.txt` pre-step (which runs before the
     entry point and would otherwise hard-fail the job) never triggers.
     `entrypoint_tier1.py` instead installs every other pin from
     `requirements.txt` verbatim (`vllm==0.5.0`, `vllm-flash-attn==2.5.9`,
     `transformers==4.44.2`, etc.) and leaves only `litellm` unpinned.
     Nothing in `pipeline/` or `scripts/` actually calls litellm's API
     unless the `"llamaguard2"` jailbreak-eval methodology is selected,
     which no config in this fork uses — litellm is only imported at
     module level in `pipeline/submodules/evaluate_jailbreak.py` and must
     resolve for that import to succeed, nothing more. The original
     `requirements.txt` file itself is untouched in the repo.
  2. **Missing dependencies**: `requirements.txt` never listed `mmengine`,
     `deep-translator`, or `jsonpickle` at all, even though
     `dataset/load_dataset.py`, `pipeline/run_pipeline.py`, and
     `scripts/multi_test.py` import them. Pinned versions of these three
     are installed alongside the `requirements.txt`-derived set above.
  3. **Path-naming mismatch (pre-existing bug in the official repo, not
     introduced here)**: `pipeline/run_pipeline.py`'s
     `select_and_save_direction()` saves the selected direction as
     `<artifact_path>/direction_ablation.pt`, but `scripts/multi_test.py`
     loads `<artifact_path>/direction.pt` (no `_ablation` suffix) — see
     `pipeline/run_pipeline.py:139` vs. `scripts/multi_test.py:73,78`.
     Neither script bridges this itself. `entrypoint_tier1.py` copies
     `direction_ablation.pt` → `direction.pt` after extraction, which is
     exactly the manual step a human running these two scripts back-to-back
     by hand would have needed. No vector values are altered by the copy.
- `launch_tier1.py` — SageMaker job launcher (one job per source language,
  `wait=False` so they run in parallel). Reuses the IAM role, region and
  `PyTorch` estimator conventions already established in
  `~/Downloads/safety-layers-repro/sagemaker/` — an unrelated reproduction
  of a *different* paper ("Safety Layers in Aligned LLMs", Li et al. 2025)
  that happened to already have AWS/SageMaker infra configured on this
  machine. Only the HF token file from that setup is reused (a generic HF
  token, not specific to that repo); the Anthropic key configured there is
  **not** used — this paper's own method judges compliance with WildGuard,
  not an LLM judge, and substituting one would be an unrequested methodology
  change. Also stages a pruned copy of the repo for upload (excludes `.git`,
  the 581M checked-in `output/` directory, and — per the requirements.txt
  issue above — `requirements.txt` itself), since SageMaker's `source_dir`
  tars whatever directory it's pointed at, and none of those three are
  needed by the job.
- `CHANGES.md` — this file.

## Model switch: Qwen2.5-7B-Instruct → gemma-2b-it

Original plan was Qwen2.5-7B-Instruct on `ml.g6e.xlarge` (single 48GB L40S
GPU). In practice:

- The job needs both the target model and WildGuard (~7B) resident on the
  GPU simultaneously — the official `evaluate_jailbreak()` code doesn't
  unload the target model before loading WildGuard for scoring — roughly
  28-30GB combined for Qwen2.5-7B-Instruct + WildGuard in bf16.
- `ml.g6e.xlarge` hit `"Training job waiting for capacity"` (AWS's own
  physical instance-fleet availability for that type in `us-east-1`, not a
  quota problem — confirmed via `service-quotas:ListServiceQuotas`, which
  showed `1.0` for every `g6e` and `g5` size checked).
- Switching the target model to `google/gemma-2b-it` (~4-5GB in bf16, one
  of the paper's own benchmarked models) drops the combined footprint with
  WildGuard to ~18-19GB, fitting comfortably on `ml.g5.xlarge` (single 24GB
  A10G), which had live capacity when tested.
- Confirmed HF access to `google/gemma-2b-it` under the configured token
  before relaunching (`gated: manual`, full model metadata returned, no
  access error).

## Why a separate project folder

Per instruction, this replication lives in its own clone
(`mr-tier1-sagemaker/repo`, a fresh `git clone` of the user's fork
`silu101/Multilingual-Refusal`) rather than inside `safety-layers-repro/`,
so the two reproductions stay independent. Only AWS credentials, the IAM
execution role, and the HF token file are shared between them (all
credential/config reuse, no code or data reuse).

## What was intentionally NOT changed

- No changes to `pipeline/submodules/*.py`, `pipeline/utils/*.py`,
  `pipeline/model_utils/*.py`, `dataset/*.py`, or `evaluators/*.py` — the
  refusal-direction extraction, selection, ablation-hook, and WildGuard
  scoring logic all run exactly as authored upstream.
- No change to any hyperparameter in the existing per-language configs
  (n_train=128, n_val=32, n_test=572 via `PolyRefuse/harmful_test_translated_*.json`,
  kl_threshold=0.1, ablate_kl_threshold=0.2, batch_size=64, max_new_tokens=512,
  random_seed=1, top_n=1).
- The `select_direction` KL-filtered layer/position sweep is re-run in full
  for every source language (en/de/zh/th), not shortcut using the
  `direction_metadata_ablation.json` already committed for de/zh/th from a
  prior Qwen run — re-selecting is the official method, not overhead to
  skip.
