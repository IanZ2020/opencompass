from mmengine.config import read_base

with read_base():
    from .datasets.collections.base_medium_llama import humaneval_datasets, hellaswag_datasets,  piqa_datasets, siqa_datasets, winogrande_datasets, ARC_e_datasets, ARC_c_datasets, mbpp_datasets, obqa_datasets, commonsenseqa_datasets, nq_datasets, triviaqa_datasets, squad20_datasets, BoolQ_datasets, gsm8k_datasets, math_datasets, mmlu_datasets, bbh_datasets, nq_retrieval_datasets, triviaqa_retrieval_datasets

from opencompass.models import PrunedllamaCausalLM
from opencompass.models import HuggingFaceCausalLM

batch=4

models = [
    # LLaMA 13B
    dict(
        type=PrunedllamaCausalLM,
        abbr='llama-7b-mha-1-32-0.25-8192Redpajama-first-local-weighted-expt-0.05-mlp-0-31-0.25-8192Redpajama-first-local-weighted-expt-0.1-recover',
        path="/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/7b-mha+mlp/llama-7b-mha-1-32-0.25-8192Redpajama-first-local-weighted-expt-0.05-mlp-0-31-0.25-8192Redpajama-first-local-weighted-expt-0.1",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=batch,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=PrunedllamaCausalLM,
        abbr='llama-13b-mha-1-40-0.25-first-local-weighted-expt-0.05-mlp-new-0-39-0.25-8192Redpajama-first-local-weighted-expt-0.1-recover',
        path="/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/13b-mha+mlp/llama-13b-mha-1-40-0.25-8192Redpajama-first-local-weighted-expt-0.05-mlp-new-0-39-0.25-8192Redpajama-first-local-weighted-expt-0.1",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=batch,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-13b-hf',
        path="pinkmanlove/llama-13b-hf",
        tokenizer_path='pinkmanlove/llama-13b-hf',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=batch,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-7b-hf',
        path="pinkmanlove/llama-7b-hf",
        tokenizer_path='pinkmanlove/llama-13b-hf',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=batch,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

#full eval
# datasets = [*humaneval_datasets, *mbpp_datasets, *hellaswag_datasets,  *piqa_datasets, *siqa_datasets, *winogrande_datasets, *ARC_e_datasets, *ARC_c_datasets, *obqa_datasets, *commonsenseqa_datasets, *nq_datasets, *nq_retrieval_datasets, *triviaqa_datasets, *triviaqa_retrieval_datasets, *squad20_datasets, *BoolQ_datasets, *gsm8k_datasets,  *math_datasets, *mmlu_datasets, *bbh_datasets]

#short eval
# datasets = [*hellaswag_datasets,  *piqa_datasets, *siqa_datasets, *winogrande_datasets, *ARC_e_datasets, *ARC_c_datasets, *obqa_datasets, *commonsenseqa_datasets, *nq_datasets, *triviaqa_datasets, *squad20_datasets, *BoolQ_datasets, *mmlu_datasets]

datasets = [*nq_retrieval_datasets, *nq_datasets]
