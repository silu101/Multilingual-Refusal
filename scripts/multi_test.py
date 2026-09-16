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


class AmazonTranslateClient:
    """Drop-in replacement for deep_translator.GoogleTranslator exposing
    the same .translate(text) -> str interface, backed by Amazon Translate
    instead. See sagemaker_tier1/CHANGES.md for why: GoogleTranslator's
    underlying free/unofficial endpoint was persistently returning
    TooManyRequests from this AWS account (confirmed even with
    self-throttling under its stated 5 req/s limit -- likely a
    datacenter-IP-range-level block, not our own request rate). Amazon
    Translate is a paid, official AWS service reachable with the same
    SageMaker execution role credentials already in use, no new API key
    needed, and has a much higher default quota (100 TPS) than Google's
    free tier. Known gap: it does not support Yoruba
    (UnsupportedLanguagePairException, confirmed locally) -- 'yo' is
    excluded from this replication round rather than silently dropped or
    worked around; see CHANGES.md."""

    # Amazon Translate's synchronous TranslateText has a 10,000-byte (UTF-8)
    # limit per request; some target languages (th, zh, ar, ja, ko) use
    # multi-byte characters where the repo's existing 4999-*character*
    # truncation (see the call sites below) could still exceed that. This
    # truncates by encoded byte length, with margin, on top of the
    # existing character-based truncation.
    MAX_BYTES = 9000

    def __init__(self, source_lang: str, target_lang: str = 'en', region_name: str = 'us-east-1'):
        import boto3
        self.client = boto3.client('translate', region_name=region_name)
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate(self, text: str) -> str:
        encoded = text.encode('utf-8')
        if len(encoded) > self.MAX_BYTES:
            text = encoded[:self.MAX_BYTES].decode('utf-8', errors='ignore')
        response = self.client.translate_text(
            Text=text,
            SourceLanguageCode=self.source_lang,
            TargetLanguageCode=self.target_lang,
        )
        return response['TranslatedText']


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
    
    # Switched from GoogleTranslator to AmazonTranslateClient -- see its
    # docstring above and sagemaker_tier1/CHANGES.md. Amazon Translate uses
    # plain 'zh' directly (no zh-CN special-casing needed, unlike Google).
    translator = AmazonTranslateClient(source_lang=cfg.lang, target_lang='en')
    
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
        # Real error visibility + retry-with-backoff + self-throttling (see
        # sagemaker_tier1/CHANGES.md): the original loops below caught
        # every exception and silently replaced it with the literal string
        # "Translation failed", which then got fed straight to WildGuard as
        # if it were the model's response -- a 100% translation failure
        # rate (confirmed on job mr-tier1-en-2026-09-15-04-14-13-875) would
        # silently produce a meaningless "all safe" WildGuard score instead
        # of an error. Root cause (confirmed on job
        # mr-tier1-en-2026-09-15-05-07-12-047, once errors were no longer
        # swallowed): deep_translator.GoogleTranslator's underlying (free,
        # unofficial) endpoint enforces "5 requests per second and up to
        # 200k requests per day" and was returning TooManyRequests on every
        # call. The library's own suggested fix, translate_batch(), does
        # NOT help -- checked its source locally: it's just a Python-level
        # loop calling .translate() once per item, i.e. the exact same
        # number of HTTP requests. translate_with_retry() below instead
        # self-throttles to stay under that stated per-second limit and
        # backs off substantially longer than before specifically on
        # TooManyRequests (the error message says "wait and try again
        # later", not "try again in 2 seconds"). Same fallback behavior on
        # final failure as originally (falls back to the untranslated
        # response).
        _last_translate_call = [0.0]
        # Now backed by Amazon Translate (see AmazonTranslateClient above),
        # not Google -- its default quota is 100 TPS/account, far higher
        # than Google's free-tier 5 req/s, so this only needs to keep us
        # comfortably under that (also boto3 has its own built-in retry
        # handling for AWS API throttling on top of this).
        min_gap_seconds = 0.02  # ~50 req/s, well under Amazon Translate's 100 TPS default quota
        # Circuit breaker: if the endpoint is persistently blocking us
        # (e.g. a shared-NAT-IP daily quota already exhausted by other AWS
        # tenants, not just our own request rate), retrying every single
        # item is pure wasted GPU-hours with no chance of succeeding. After
        # this many consecutive items exhaust every retry, stop retrying
        # for the rest of the run and fall back immediately.
        _consecutive_full_failures = [0]
        CIRCUIT_BREAKER_THRESHOLD = 5

        def translate_with_retry(text, max_attempts=3, backoff_seconds=5.0):
            if _consecutive_full_failures[0] >= CIRCUIT_BREAKER_THRESHOLD:
                return None
            last_err = None
            for attempt in range(1, max_attempts + 1):
                elapsed = time.time() - _last_translate_call[0]
                if elapsed < min_gap_seconds:
                    time.sleep(min_gap_seconds - elapsed)
                try:
                    result = translator.translate(text)
                    _last_translate_call[0] = time.time()
                    _consecutive_full_failures[0] = 0
                    return result
                except Exception as e:
                    _last_translate_call[0] = time.time()
                    last_err = e
                    if attempt < max_attempts:
                        time.sleep(backoff_seconds * attempt)
            _consecutive_full_failures[0] += 1
            print(f"Translation failed after {max_attempts} attempts: {type(last_err).__name__}: {last_err}")
            if _consecutive_full_failures[0] >= CIRCUIT_BREAKER_THRESHOLD:
                print(f"{CIRCUIT_BREAKER_THRESHOLD} consecutive translation failures -- "
                      f"assuming the endpoint is blocking this run and disabling further retries "
                      f"(falling back to untranslated responses for the remainder).")
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

    # The existing "clear the gpu" call above (before the first
    # evaluate_jailbreak()) only covers generation's leftover memory --
    # there was no equivalent clear between this evaluate_jailbreak() call
    # and the next one, even though each one runs a full WildGuard scoring
    # pass over test_sample_size completions. Confirmed necessary: job
    # mr-tier1-en-2026-09-16-11-23-36-300 OOM'd inside WildGuard's own
    # forward pass (modeling_mistral.py) at test_sample_size=250, between
    # the ablation and baseline evaluate_jailbreak() calls.
    torch.cuda.empty_cache()

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
        torch.cuda.empty_cache()  # same reasoning as the clear above, between this evaluate_jailbreak() call and the previous one
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