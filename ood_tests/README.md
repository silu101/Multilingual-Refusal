# OOD evaluation

Code for two of the three OOD shifts in "OOD Generalization Tests for
Mechanistic LLM Safety" (§2 and §3), applied to this repo's
refusal-direction ablation method (Wang et al. — the Universal Refusal
Direction paper this fork's Tier-1 replication targets). Both extract the
refusal direction from in-distribution data and measure how much the
ablation effect holds up on prompts/attacks it never saw.

**Status: code written and syntax-checked, not yet run end-to-end.** No
SageMaker job has been launched for this yet — see "Before running" below
for what to confirm first. The pure-data pieces (splitting, dedup,
fetching) have been run and verified locally against real data; the model
side (`run_ood_eval.py`) has not.

## Semantic-content OOD (§2, Table 2)

| Dataset | Role | Source |
|---|---|---|
| AdvBench (pure, split 80/20) | Extraction train/val | Already shipped (`dataset/processed/advbench.json`) |
| HarmBench test, deduped against AdvBench | Baseline (in-family) eval | Already shipped, deduped on the fly |
| BeaverTails | OOD eval | Fetched (`ood_tests/fetch_ood_datasets.py`) |
| Anthropic red-team-attempts (community mirror) | OOD eval | Fetched |
| OpenAI Moderation | OOD eval | Fetched |
| Aegis 2.0 | OOD eval | Fetched |
| SimpleSafetyTests | OOD eval | Fetched |

## Attack/jailbreak-family OOD (§3, Table 3)

Same underlying idea, but the thing held out is the *attack transformation*
applied to a goal, not the goal's semantic category:

| Resource | What's held out | Source |
|---|---|---|
| JailBreakLLMs-wrapped | Held-out AdvBench goals wrapped in in-the-wild DAN/persona templates the direction never saw | Fetched (`ood_tests/fetch_attack_datasets.py`) |
| WildJailbreak (adversarial) | A different, pre-composed adversarial-wrapper distribution | Fetched, gated dataset (see below) |

**Only implemented for the two Table 3 resources that are actual
datasets.** GCG, AutoDAN, and PAIR are not datasets — they're live attack
*algorithms* (GCG: gradient-based token-suffix search; AutoDAN: genetic
search; PAIR: an attacker LLM iteratively refining prompts against live
target-model feedback) that would each need their own codebase actually
running against the target model, with real additional compute cost per
prompt (GCG in particular is expensive — many forward+backward passes per
optimized suffix). Building those is a substantially bigger, separate
effort, not attempted here.

`jailbreakllms_wrapped` is deliberately goal-controlled: it reuses the same
AdvBench goals used for extraction (per the OOD paper's own framing in §3:
"The harmful goal and its jailbreak wrapper must be stored separately"),
just wrapped in a template family the direction never saw — so any
compliance drop here isolates the effect of the *wrapper*, not a
confounding change in *what's being asked*. Of the 1405 in-the-wild
templates in `TrustAIRLab/in-the-wild-jailbreak-prompts`, only 94 contain
an explicit `[INSERT PROMPT HERE]` insertion point usable as a single-turn
prompt via direct substitution — the rest are "persona-priming" templates
meant to be sent as a first conversational turn, with the actual harmful
ask following in a second turn, which would need multi-turn generation
support this repo's pipeline doesn't have
(`model_base.generate_completions()` is single-turn only). Restricted to
those 94, disclosed here rather than silently narrowed.

`wildjailbreak_adversarial` (`allenai/wildjailbreak`, `eval` config,
`data_type == "adversarial_harmful"`) is **gated** on the HF Hub
(self-serve "auto" gate, same kind as `google/gemma-2b-it` and
`allenai/wildguard` during the Tier-1 replication) — confirmed accessible
under the currently-configured HF token without needing a fresh accept
step, but flagging in case a different token/account is used to run this.

## Not implemented at all

The full utility suite (§4) — only substring-matching/WildGuard compliance
is measured throughout `ood_tests/`; MMLU/GSM8K/BBH/TruthfulQA/XSTest/
WildJailbreak-benign are not. `run_pipeline()`'s own `skip_eval_harness`
flag is set `True` in `run_ood_eval.py`'s config for the same reason it was
during the Tier-1 replication — MMLU checking needs a separate `lm_eval`
install this fork's `requirements.txt` doesn't provide, and it's orthogonal
to this experiment's actual target.

## Why pure AdvBench, not the repo's default `harmful` split

The official `dataset/splits/harmful_train.json`/`harmful_val.json` (what
`pipeline.run_pipeline` uses unless overridden) are a mix of AdvBench +
MaliciousInstruct + TDC2023, per the Wang et al. paper's own methodology
(confirmed directly: `harmful_val.json`'s one sample is textually identical
to `harmbench_val.json`'s). Extracting the direction on that mix and then
calling MaliciousInstruct "OOD" would be circular — it was already in
training. `ood_tests/split_advbench.py` splits `advbench.json` alone (520
items, no official split) into `dataset/splits/harmful_advbench_{train,val}.json`
(416/104, seed 1), and `dataset/load_dataset.py` gained a
`'harmful_advbench'` harmtype to load it — the only change to code outside
`ood_tests/` this required. `run_ood_eval.py`'s config sets
`cfg.harmtype_2 = 'harmful_advbench'` to use it for extraction.

## Why HarmBench gets deduped against AdvBench

The OOD paper's own caution (Table 2): HarmBench "overlaps with AdvBench,
so deduplicate before cross-dataset OOD." `ood_tests/dedup.py` does a
normalized-text similarity check (stdlib `difflib`, threshold 0.85, not
exact-match only, to catch paraphrased overlaps) and
`run_ood_eval.py.ensure_harmbench_deduped()` runs it automatically on first
use, writing `dataset/processed/harmbench_test_deduped.json`.

## Dataset provenance notes worth knowing

- **Anthropic red-team-attempts**: fetched from `gohsyi/red-team-attempts`
  on the HF Hub — a public, non-gated **community mirror** of Anthropic's
  original release (Ganguli et al., 2022), not an official Anthropic HF
  dataset. Flagging this since provenance matters for a research
  replication; verified the dataset exists and its schema before writing
  code against it, but did not independently verify it's a faithful,
  unmodified copy of the original release.
- This dataset also skews toward graphic offensive-language/harassment
  content rather than AdvBench's actionable-technical-harm style, and
  contains many entries that are just insults/slurs rather than a request
  for harmful assistance. `fetch_ood_datasets.py`'s
  `fetch_anthropic_redteam()` filters to
  `task_descripton_harmlessness_score >= 0.5` (the red-teamer's own stated
  intent scored as more clearly harmful) as a rough quality filter, and
  uses only the first human turn (`prompt[0]['content']`) as the
  "instruction," not the full multi-turn transcript or the model's own
  (often extremely graphic) prior responses.
- **maliciousinstruct**: not used by this OOD experiment (it's part of the
  contaminated default training mix, see above), but worth noting for
  anyone touching `dataset/load_dataset.py`: `PROCESSED_DATASET_NAMES`
  lists it as `"maliciousinstruct"` but the actual file on disk is
  `dataset/processed/malicious_instruct.json` (underscore) — a
  pre-existing mismatch in the official repo, not introduced here.

## Before running

1. **Install `datasets`** (not in `requirements.txt` or
   `sagemaker_tier1/requirements_fixed.txt`) — needed by
   `fetch_ood_datasets.py` and `fetch_attack_datasets.py`. Everything else
   needed (`mmengine`, `torch`, `transformers`, etc.) is already covered by
   the Tier-1 replication's dependency set.
2. Decide model, sample size, and instance sizing — this hasn't been run
   yet, so there's no measured GPU-hour or cost number for it the way
   Tier-1 has. Expect it to need the same combined VRAM budget as Tier-1
   (target model + WildGuard resident simultaneously — see
   `sagemaker_tier1/CHANGES.md`'s "Model switch" section for that
   reasoning), scaled by however many OOD-dataset prompts you evaluate.

## Running it

```bash
# One-time setup
python -m ood_tests.split_advbench
python -m ood_tests.fetch_ood_datasets            # semantic-content OOD sets; add --limit N for a quick smoke test
python -m ood_tests.fetch_attack_datasets         # attack-family OOD sets; add --n-wrapped N / --limit N for a quick smoke test

# Main run -- semantic-content OOD (default dataset list)
python -m ood_tests.run_ood_eval --model_path google/gemma-2b-it \
    --n_ood_sample 100 --wildguard_batch_size 8

# Attack-family OOD -- same script, just point it at the attack datasets instead
python -m ood_tests.run_ood_eval --model_path google/gemma-2b-it \
    --ood_datasets harmbench_test_deduped jailbreakllms_wrapped wildjailbreak_adversarial \
    --n_ood_sample 100 --wildguard_batch_size 8
```

Results land in `ood_tests/runs/<model_alias>/ood_results.json`: per-dataset
`baseline_compliance`, `ablated_compliance`, and `lift` (ablated minus
baseline), plus each OOD dataset's lift expressed as a ratio against the
HarmBench (in-family) baseline lift — the direct measure of how much the
ablation effect degrades out-of-distribution.
