from mmengine.config import read_base

with read_base():
    from .datasets.collections.base_medium_llama import humaneval_datasets, hellaswag_datasets,  piqa_datasets, siqa_datasets, winogrande_datasets, ARC_e_datasets, ARC_c_datasets, mbpp_datasets, obqa_datasets, commonsenseqa_datasets, nq_datasets, triviaqa_datasets, squad20_datasets, BoolQ_datasets, gsm8k_datasets, math_datasets, mmlu_datasets, bbh_datasets, nq_retrieval_datasets, triviaqa_retrieval_datasets

from opencompass.models import PrunedllamaCausalLM
from opencompass.models import HuggingFaceCausalLM

batch=4

paths = [
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/13b-mha+mlp/llama-13b-mha-1-40-0.25-8192Redpajama-first-local-weighted-expt-0.05-mlp-new-0-39-0.25-8192Redpajama-first-local-weighted-expt-0.1",
    "/home/zhangyihan/LMFlow/prune_log/13b-mha+mlp/llama-13b-mha-1-40-0.25-8192Redpajama-first-local-weighted-expt-0.05-mlp-new-0-39-0.25-8192Redpajama-first-local-weighted-expt-0.1",
    "/home/zhangyihan/LMFlow/prune_log/2.7b-mha+mlp/llama-2.7b-mlp-0-31-0.125-8192red-first-local-weighted-expt-0.1-2.7b-mha-1-32-0.125-8192red-first-local-weighted-expt-0.05",
    "/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-mlp-0-31-0.25-8192red-second-local-weighted-expt-0.1-7b-mha-1-32-0.25-8192red-second-local-weighted-expt-0.05",
    "/home/zhangyihan/LMFlow/prune_log/13b-mha+mlp/llama-13b-mha-1-40-0.25-8192Redpajama-second-local-weighted-expt-0.05-mlp-new-0-39-0.25-8192Redpajama-second-local-weighted-expt-0.1",
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/2.7b-mha+mlp/llama-2.7b-mlp-0-31-0.125-8192red-first-local-weighted-expt-0.1-2.7b-mha-1-32-0.125-8192red-first-local-weighted-expt-0.05"]

path_recover = [paths[0], paths[-1]]
models = [
    # LLaMA 13B
    dict(
        type=PrunedllamaCausalLM,
        abbr=path.split('/')[-1]+'-recover',
        path=path,
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              unk_token="<unk>",
                              bos_token="<s>",
                              eos_token="</s>",
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=batch,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ) for path in path_recover]

paths = paths[1:-1]
models += [dict(
        type=PrunedllamaCausalLM,
        abbr=path.split('/')[-1],
        path=path,
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              unk_token="<unk>",
                              bos_token="<s>",
                              eos_token="</s>",
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=batch,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ) for path in paths]


models += [dict(
        type=HuggingFaceCausalLM,
        abbr='Sheared-LLaMA-2.7B',
        path='princeton-nlp/Sheared-LLaMA-2.7B',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              unk_token="<unk>",
                              bos_token="<s>",
                              eos_token="</s>",
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=batch,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        abbr='pythia-12b',
        path='EleutherAI/pythia-12b',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              unk_token="<unk>",
                              bos_token="<s>",
                              eos_token="</s>",
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=batch,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),]


#full eval
# datasets = [*humaneval_datasets, *mbpp_datasets, *hellaswag_datasets,  *piqa_datasets, *siqa_datasets, *winogrande_datasets, *ARC_e_datasets, *ARC_c_datasets, *obqa_datasets, *commonsenseqa_datasets, *nq_datasets, *nq_retrieval_datasets, *triviaqa_datasets, *triviaqa_retrieval_datasets, *squad20_datasets, *BoolQ_datasets, *gsm8k_datasets,  *math_datasets, *mmlu_datasets, *bbh_datasets]

#short eval
datasets = [*hellaswag_datasets,  *piqa_datasets, *siqa_datasets, *winogrande_datasets, *ARC_e_datasets, *ARC_c_datasets, *obqa_datasets, *commonsenseqa_datasets, *nq_datasets, *triviaqa_datasets, *squad20_datasets, *BoolQ_datasets]

