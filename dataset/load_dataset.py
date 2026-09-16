import os
import json

dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

# SPLITS = ['train', 'val', 'test']
# 'harmful_advbench' added for ood_tests/ (see ood_tests/README.md and
# ood_tests/split_advbench.py): a pure-AdvBench-only harmtype, distinct from
# the default 'harmful' (which mixes AdvBench + MaliciousInstruct + TDC2023,
# and would contaminate any later OOD test against MaliciousInstruct
# specifically). Existing HARMTYPES/PROCESSED_DATASET_NAMES entries and
# their files are untouched.
HARMTYPES = ['harmless', 'harmful', 'harmful_advbench', 'xstest', "or_bench_hard", "oktest", 'ok_or', 'xstest_safe', 'xstest_unsafe', "jailbreakbench"] # TODO: harmtype cannot actually handle xstest and oktest yet

SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, 'splits/{harmtype}_{split}.json')

SPLIT_DATASET_MULTI_FILENAME = os.path.join(dataset_dir_path, 'splits_multi/{harmtype}_{split}_translated_{lang}.json')

# beavertails, anthropic_redteam, openai_moderation, aegis2, simplesafetytests
# added for ood_tests/fetch_ood_datasets.py's outputs (see
# ood_tests/README.md). Also note: "maliciousinstruct" below has never
# matched its actual file on disk, dataset/processed/malicious_instruct.json
# (underscore) -- a pre-existing inconsistency, not introduced here; load
# that one directly by filename rather than via load_dataset("maliciousinstruct").
PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", "jailbreakbench", "strongreject", "alpaca", "over_refusal", "xstest", "oktest", "oktest_100", "xstest_unsafe",  "xstest_safe",
                            "beavertails", "anthropic_redteam", "openai_moderation", "aegis2", "simplesafetytests",
                            "harmbench_test_deduped",
                            "jailbreakllms_wrapped", "wildjailbreak_adversarial"]

def load_dataset_split(harmtype: str, split: str, lang: str='en', instructions_only: bool=False):
    assert harmtype in HARMTYPES
    # assert split in SPLITS

    if lang == 'en':
        file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)
    else:
        file_path = SPLIT_DATASET_MULTI_FILENAME.format(harmtype=harmtype, split=split, lang=lang)

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]
 
    return dataset

# def load_dataset_split_multilingual(harmtype: str, split: str, lang: str, instructions_only: bool=False):
#     assert harmtype in HARMTYPES
#     assert split in SPLITS

#     file_path = SPLIT_DATASET_MULTI_FILENAME.format(harmtype=harmtype, split=split, lang=lang)

#     with open(file_path, 'r') as f:
#         dataset = json.load(f)

#     if instructions_only:
#         dataset = [d['instruction_translated'] for d in dataset]

#     return dataset