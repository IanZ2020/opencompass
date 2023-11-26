from mmengine.config import read_base

with read_base():
    from .datasets.collections.base_medium_llama import humaneval_datasets, hellaswag_datasets,  piqa_datasets, siqa_datasets, winogrande_datasets, ARC_e_datasets, ARC_c_datasets, mbpp_datasets, obqa_datasets, commonsenseqa_datasets, nq_datasets, triviaqa_datasets, squad20_datasets, BoolQ_datasets, gsm8k_datasets, math_datasets, mmlu_datasets, bbh_datasets, nq_retrieval_datasets, triviaqa_retrieval_datasets

from opencompass.models import PrunedllamaCausalLM
from opencompass.models import HuggingFaceCausalLM

batch=16

paths = [
    "/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-bookcorpus1024-mlp-0-31-0.25-first-local-weighted-expt-0.1-7b-mha-1-32-0.25-first-local-weighted-expt-0.05",
    "/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-github1024-mlp-0-31-0.25-first-local-weighted-expt-0.1-7b-mha-1-32-0.25-first-local-weighted-expt-0.05",
    "/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-stack1024-mlp-0-31-0.25-first-local-weighted-expt-0.1-7b-mha-1-32-0.25-first-local-weighted-expt-0.05",
    "/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-wiki1024-mlp-0-31-0.25-first-local-weighted-expt-0.1-7b-mha-1-32-0.25-first-local-weighted-expt-0.05",
    "/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-16384red-mlp-0-31-0.25-first-local-weighted-expt-0.1-7b-mha-1-32-0.25-first-local-weighted-expt-0.05",
    "/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-8192red-mlp-0-31-0.3-first-local-weighted-expt-0.1-7b-mha-1-32-0.3-first-local-weighted-expt-0.05",
    ]


models = [dict(
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

paths = [
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-8192red-mlp-0-31-0.3-first-local-weighted-expt-0.1-7b-mha-1-32-0.3-first-local-weighted-expt-0.05/checkpoint-50",
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-8192red-mlp-0-31-0.3-first-local-weighted-expt-0.1-7b-mha-1-32-0.3-first-local-weighted-expt-0.05/checkpoint-100",
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-8192red-mlp-0-31-0.3-first-local-weighted-expt-0.1-7b-mha-1-32-0.3-first-local-weighted-expt-0.05/checkpoint-150",
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-8192red-mlp-0-31-0.3-first-local-weighted-expt-0.1-7b-mha-1-32-0.3-first-local-weighted-expt-0.05/checkpoint-200",
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-8192red-mlp-0-31-0.3-first-local-weighted-expt-0.1-7b-mha-1-32-0.3-first-local-weighted-expt-0.05/checkpoint-250",
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-8192red-mlp-0-31-0.3-first-local-weighted-expt-0.1-7b-mha-1-32-0.3-first-local-weighted-expt-0.05/checkpoint-300",
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-8192red-mlp-0-31-0.3-first-local-weighted-expt-0.1-7b-mha-1-32-0.3-first-local-weighted-expt-0.05/checkpoint-350",
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-8192red-mlp-0-31-0.3-first-local-weighted-expt-0.1-7b-mha-1-32-0.3-first-local-weighted-expt-0.05/checkpoint-400",
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-8192red-mlp-0-31-0.3-first-local-weighted-expt-0.1-7b-mha-1-32-0.3-first-local-weighted-expt-0.05/checkpoint-450",
    "/home/zhangyihan/LMFlow/output_models/pruning_study_pretraining_red_lr1e-4_full/home/zhangyihan/LMFlow/prune_log/7b-mha+mlp/llama-7b-8192red-mlp-0-31-0.3-first-local-weighted-expt-0.1-7b-mha-1-32-0.3-first-local-weighted-expt-0.05/checkpoint-500",
    ]

models += [dict(
        type=PrunedllamaCausalLM,
        abbr=path.split('/')[-2]+path.split('/')[-1],
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

#full eval
# datasets = [*humaneval_datasets, *mbpp_datasets, *hellaswag_datasets,  *piqa_datasets, *siqa_datasets, *winogrande_datasets, *ARC_e_datasets, *ARC_c_datasets, *obqa_datasets, *commonsenseqa_datasets, *nq_datasets, *nq_retrieval_datasets, *triviaqa_datasets, *triviaqa_retrieval_datasets, *squad20_datasets, *BoolQ_datasets, *gsm8k_datasets,  *math_datasets, *mmlu_datasets, *bbh_datasets]

#short eval
datasets = [*hellaswag_datasets,  *piqa_datasets, *siqa_datasets, *winogrande_datasets, *ARC_e_datasets, *ARC_c_datasets, *obqa_datasets, *commonsenseqa_datasets, *nq_datasets, *triviaqa_datasets, *squad20_datasets, *BoolQ_datasets]

