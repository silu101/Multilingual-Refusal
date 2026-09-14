# Changes made for the SageMaker Tier-1 replication

Everything in this folder is **new**, added to run the official, unmodified
`pipeline/` and `scripts/` code on Amazon SageMaker. Nothing outside this
folder was changed except the one addition noted below. This file exists
because the person requesting this replication asked that any deviation
from the official repo be recorded rather than assumed.

## Scope

Tier-1 replication (English/German/Chinese/Thai refusal-direction
extraction, cross-lingual ablation across all 14 PolyRefuse languages),
**Qwen2.5-7B-Instruct only** (gemma-2-9b-it deferred by user decision).
Budget ceiling: $100 total AWS spend.

## Files added

- `en.yaml` — a run config for `source_lang=en`. This fork ships per-language
  configs for de/ja/ko/ru/th/yo/zh under `pipeline/runs/Qwen2.5-7B-Instruct/<lang>/`,
  but none for English. `en.yaml` is byte-identical to those in every field
  except `lang`, `source_lang`, and `artifact_path` — it does not introduce
  any new hyperparameters or method changes, it just fills the missing
  English entry in the existing per-language-config convention.
- `entrypoint_tier1.py` — SageMaker training entry point. Calls
  `pipeline.run_pipeline.run_pipeline()` and `scripts.multi_test.main()`
  **unmodified**, looped over target languages. See its module docstring
  for the full step-by-step. Two things worth flagging:
  1. **Missing dependencies**: `requirements.txt` in this fork does not list
     `mmengine`, `deep-translator`, or `jsonpickle`, even though
     `dataset/load_dataset.py`, `pipeline/run_pipeline.py`, and
     `scripts/multi_test.py` import them. The official code cannot run at
     all without them. `entrypoint_tier1.py` installs pinned versions of
     these three packages before importing anything from `pipeline/` or
     `scripts/`. No other packages are added or version-changed.
  2. **Path-naming mismatch (pre-existing bug in the official repo, not
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
  change.
- `CHANGES.md` — this file.

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
  prior run — re-selecting is the official method, not overhead to skip.
