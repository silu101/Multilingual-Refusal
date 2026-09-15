# Changes made for the SageMaker Tier-1 replication

Everything in this folder is **new**, added to run the official, unmodified
`pipeline/` and `scripts/` code on Amazon SageMaker. Nothing outside this
folder was changed except the one addition noted below. This file exists
because the person requesting this replication asked that any deviation
from the official repo be recorded rather than assumed.

## Scope (current)

Tier-1 replication (English/German/Chinese/Thai refusal-direction
extraction, cross-lingual ablation across all 14 PolyRefuse languages).
Budget ceiling: $100 total AWS spend, additionally capped at ~5 GPU-hours
total wall time by user request.

## Cost-control changes to scripts/multi_test.py

The full sweep is 4 source languages × 14 target languages = 56 (source,
target) pairs, each originally generating 3 completion variants (baseline,
ablation, activation-addition) over the full 572-prompt PolyRefuse test set
per language, then scoring every one with WildGuard — whose own evaluator
(`evaluators/wildguard.py`) scores completions **one at a time** in a plain
Python loop, not batched, and is likely the dominant cost, independent of
which model generated the text. That workload doesn't fit a 5 GPU-hour
target, so two **opt-in, backward-compatible** knobs were added to
`scripts/multi_test.py`'s `main()`:

- `cfg.test_sample_size` (int, default unset): when set, subsamples each
  language's 572-prompt test set down to this many prompts, seeded with
  `cfg.random_seed` for reproducibility. Unset (the default for anyone
  using this repo without setting it) keeps the original full 572.
- `cfg.skip_addition` (bool, default `False`): when `True`, skips
  generating and evaluating the activation-addition completions entirely
  (Figure 3 in the paper — vector addition, not ablation). This wasn't in
  Tier-1's scope to begin with (Tier-1 = Figures 1 & 2, ablation only), so
  skipping it costs nothing in coverage of what was actually asked for, and
  cuts roughly 1/3 of the generation + WildGuard-scoring work per pair.

Both default to the exact original behavior — running `scripts/multi_test.py`
normally, without setting either field, is unaffected. Only
`sagemaker_tier1/entrypoint_tier1.py`'s own CLI flags
(`--test_sample_size`, `--skip_addition`) set them, for this replication's
runs specifically. This is a real edit to official code (not just deploy
glue in `sagemaker_tier1/`), made because there was no way to hit the
stated 5-GPU-hour budget otherwise; flagging it prominently here rather
than folding it in quietly.

**Statistical-power caveat**: subsampling to N prompts means each
language's reported compliance rate has correspondingly higher variance
than the paper's own 572-prompt figures — this is a real trade-off for
fitting the budget, not a free lunch. The actual sample size used, and the
real measured GPU-hours it took, are recorded here once the smoke test
establishes real per-pair timing (rather than guessed from the Qwen-based
estimate, which doesn't directly transfer to gemma-2b-it + WildGuard).

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
     `sagemaker_tier1/requirements_fixed.txt` is a full copy of
     `requirements.txt` (`diff` the two files to confirm) with exactly two
     kinds of edits: the `litellm==1.40.9` line changed to unpinned
     `litellm`, and three packages appended that were missing entirely (see
     point 2). Nothing in `pipeline/` or `scripts/` actually calls
     litellm's API unless the `"llamaguard2"` jailbreak-eval methodology is
     selected, which no config in this fork uses — litellm is only
     imported at module level in
     `pipeline/submodules/evaluate_jailbreak.py` and must resolve for that
     import to succeed, nothing more. `entrypoint_tier1.py` installs from
     `requirements_fixed.txt`; the original `requirements.txt` file itself
     is untouched in the repo. (An earlier version of this fix hand-picked
     a subset of packages instead of using the full corrected file, and
     that subset missed `datasets` — imported by
     `pipeline/submodules/evaluate_loss.py`, itself imported by
     `pipeline/run_pipeline.py` — which only surfaced via a second failed
     job. Installing the full file avoids that whole class of bug.)
  2. **Missing dependencies**: `requirements.txt` never listed `mmengine`,
     `deep-translator`, or `jsonpickle` at all, even though
     `dataset/load_dataset.py`, `pipeline/run_pipeline.py`, and
     `scripts/multi_test.py` import them. Pinned versions of these three
     are appended to `requirements_fixed.txt`.
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
