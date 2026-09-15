import torch
import json
import os
import time
import os.path as osp
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import argparse
import random
from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook,get_all_direction_ablation_hooks
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
import mmengine
from pipeline.utils.hook_utils import add_hooks
import deepl
import sys
from deep_translator import GoogleTranslator
from tqdm import tqdm
from utils.utils import LoggerWriter


def evaluate_completions_and_save_results_for_dataset(cfg, lang, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path, f'{lang}/completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path, lang, "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path}/{lang}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)




def main(config_path):
    # auth_key = os.environ.get('DEEPL_KEY')
    # translator = deepl.Translator(auth_key)

    cfg = mmengine.Config.fromfile(config_path)
    time_stamp = datetime.now().strftime("%y%m%d_%H%M")
    
    test_lang = cfg.lang if cfg.lang != 'zh' else 'zh-CN'
    translator = GoogleTranslator(source=test_lang, target='en')
    
    model_alias = os.path.basename(cfg.model_path)
    cfg.model_alias = model_alias
    if 'artifact_path' not in cfg:
        cfg.artifact_path = os.path.join("output", cfg.model_alias, cfg.lang)
        
        
    
    logger = mmengine.MMLogger.get_instance(
        name="dissect",
        logger_name="dissect",
        log_file=osp.join(cfg.artifact_path, f"{time_stamp}.log"),
    )
    
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)

    model_base = construct_model_base(cfg.model_path)
    
    
    
    if cfg.source_lang == 'en':
        direction_ablation = torch.load(f'pipeline/runs/{model_alias}/direction.pt')
        
        # read from json file for layer
        layer = json.load(open(f'pipeline/runs/{model_alias}/direction_metadata_ablation.json'))['layer'][0]
    else:
        direction_ablation = torch.load(f'pipeline/runs/{model_alias}/{cfg.source_lang}/direction.pt')
        # read from json file for layer
        layer = json.load(open(f'pipeline/runs/{model_alias}/{cfg.source_lang}/direction_metadata_ablation.json'))['layer'][0]
        
    if isinstance(direction_ablation, list):
        direction_ablation = direction_ablation[0]
    
    

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    harm_actadd_fwd_pre_hooks, harm_actadd_fwd_hooks = [], []
    # or_ablation_fwd_pre_hooks, or_ablation_fwd_hooks = [], []
    or_ablation_fwd_pre_hooks, or_ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction_ablation, 0)
    harm_actadd_fwd_pre_hooks.append((model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction_ablation, coeff=+cfg.addact_coeff)))
    
    or_ablation_harm_actadd_fwd_pre_hooks = or_ablation_fwd_pre_hooks + harm_actadd_fwd_pre_hooks
    or_ablation_harm_actadd_fwd_hooks = or_ablation_fwd_hooks + harm_actadd_fwd_hooks


    data_test = load_dataset_split('harmful', split='test', lang=cfg.lang)
    dataset_name = 'harmful'
    # data_test = load_dataset_split('jailbreakbench', split='test', lang=cfg.lang)
    # dataset_name = 'jailbreakbench'

    # --- Opt-in cost-control knobs (see sagemaker_tier1/CHANGES.md) ---
    # Both default to the ORIGINAL, unmodified behavior (full 572-prompt
    # test set, all three completion variants) for anyone using this repo
    # without setting them; only sagemaker_tier1/entrypoint_tier1.py sets
    # them, to fit a fixed compute budget.
    if cfg.get('test_sample_size'):
        import random as _random
        _random.seed(cfg.get('random_seed', 1))
        data_test = _random.sample(data_test, min(cfg.test_sample_size, len(data_test)))
    skip_addition = cfg.get('skip_addition', False)

    completions = model_base.generate_completions(data_test, fwd_pre_hooks=or_ablation_harm_actadd_fwd_pre_hooks, fwd_hooks=or_ablation_harm_actadd_fwd_hooks, max_new_tokens=512, batch_size=cfg.batch_size, system=None, translation=True if cfg.lang != 'en' else False)
    completions_baseline = model_base.generate_completions(data_test, fwd_pre_hooks=baseline_fwd_pre_hooks, fwd_hooks=baseline_fwd_hooks, max_new_tokens=512, batch_size=cfg.batch_size, system=None, translation=True if cfg.lang != 'en' else False)

    if not skip_addition:
        completions_addition = model_base.generate_completions(data_test, fwd_pre_hooks=harm_actadd_fwd_pre_hooks, fwd_hooks=harm_actadd_fwd_hooks, max_new_tokens=512, batch_size=cfg.batch_size, system=None, translation=True if cfg.lang != 'en' else False)
    else:
        completions_addition = []
    
    intervention_label = cfg.mode
    



    if cfg.lang != 'en':
        # Real error visibility + retry-with-backoff (see
        # sagemaker_tier1/CHANGES.md): the original loops below caught
        # every exception and silently replaced it with the literal string
        # "Translation failed", which then got fed straight to WildGuard as
        # if it were the model's response -- a 100% translation failure
        # rate (confirmed on job mr-tier1-en-2026-09-15-04-14-13-875) would
        # silently produce a meaningless "all safe" WildGuard score instead
        # of an error. translate_with_retry() below preserves the exact
        # same fallback behavior (falls back to the literal untranslated
        # response, same as `translation if translation else
        # response['response']` did) but retries transient failures and
        # actually logs what broke on the final failure.
        def translate_with_retry(text, max_attempts=3, backoff_seconds=2.0):
            last_err = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return translator.translate(text)
                except Exception as e:
                    last_err = e
                    if attempt < max_attempts:
                        time.sleep(backoff_seconds * attempt)
            print(f"Translation failed after {max_attempts} attempts: {type(last_err).__name__}: {last_err}")
            return None

    # translate back to English and save and eval
        for response in tqdm(completions):
            # response['instruction'] = translator.translate(response['instruction'], target_lang='en')
            # response['response_translated'] = translator.translate_text(response['response'], target_lang='en-us').text
            if len (response['response']) >= 5000:
                response['response'] = response['response'][:4999]
            translation = translate_with_retry(response['response'])
            response['response_translated'] = translation if translation else response['response']


        for response in tqdm(completions_baseline):
            # response['instruction'] = translator.translate(response['instruction'], target_lang='en')
            # response['response_translated'] = translator.translate_text(response['response'], target_lang='en-us').text
            if len (response['response']) >= 5000:
                response['response'] = response['response'][:4999]
            translation = translate_with_retry(response['response'])
            response['response_translated'] = translation if translation else response['response']

        for response in tqdm(completions_addition):
            # response['instruction'] = translator.translate(response['instruction'], target_lang='en')
            # response['response_translated'] = translator.translate_text(response['response'], target_lang='en-us').text
            if len (response['response']) >= 5000:
                response['response'] = response['response'][:4999]
            translation = translate_with_retry(response['response'])
            response['response_translated'] = translation if translation else response['response']

    if not os.path.exists(os.path.join(cfg.artifact_path, 'completions')):
            os.makedirs(os.path.join(cfg.artifact_path, 'completions'))
    with open(f'{cfg.artifact_path}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)

    with open(f'{cfg.artifact_path}/completions/{dataset_name}_baseline_completions.json', "w") as f:
        json.dump(completions_baseline, f, indent=4)
        
    if not skip_addition:
        with open(f'{cfg.artifact_path}/completions/{dataset_name}_{intervention_label}_addition_completions.json', "w") as f:
            json.dump(completions_addition, f, indent=4)

    # clear the gpu 
    torch.cuda.empty_cache()

    evaluation = evaluate_jailbreak(
            completions=completions,
            methodologies=cfg.jailbreak_eval_methodologies,
            evaluation_path=os.path.join(cfg.artifact_path, "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
            translation=True if cfg.lang != 'en' else False,
            cfg=cfg,
            logger=logger
        )

    with open(f'{cfg.artifact_path}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)
        
        
    evaluation = evaluate_jailbreak(
            completions=completions_baseline,
            methodologies=cfg.jailbreak_eval_methodologies,
            evaluation_path=os.path.join(cfg.artifact_path, "completions", f"{dataset_name}_baseline_evaluations.json"),
            translation=True if cfg.lang != 'en' else False,
            cfg = cfg,
            logger=logger
            
        )

    with open(f'{cfg.artifact_path}/completions/{dataset_name}_baseline_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)
        
        
    
    if not skip_addition:
        evaluation = evaluate_jailbreak(
                completions=completions_addition,
                methodologies=cfg.jailbreak_eval_methodologies,
                evaluation_path=os.path.join(cfg.artifact_path, "completions", f"{dataset_name}_{intervention_label}_addition_evaluations.json"),
                translation=True if cfg.lang != 'en' else False,
                cfg = cfg,
                logger=logger
            )
        with open(f'{cfg.artifact_path}/completions/{dataset_name}_{intervention_label}_addition_evaluations.json', "w") as f:
            json.dump(evaluation, f, indent=4)
    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/cfg.yaml')
    args = parser.parse_args()
    config_path = args.config
    
    main(config_path)