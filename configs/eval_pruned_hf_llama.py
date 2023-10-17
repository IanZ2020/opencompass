from mmengine.config import read_base

with read_base():
    from .datasets.collections.base_medium_llama import humaneval_datasets, hellaswag_datasets,  piqa_datasets, siqa_datasets, winogrande_datasets, ARC_e_datasets, ARC_c_datasets, mbpp_datasets, obqa_datasets, commonsenseqa_datasets, nq_datasets, triviaqa_datasets, squad20_datasets, BoolQ_datasets, gsm8k_datasets, math_datasets, mmlu_datasets, bbh_datasets, nq_retrieval_datasets, triviaqa_retrieval_datasets

from opencompass.models import PrunedllamaCausalLM

models = [
    # LLaMA 13B
    dict(
        type=PrunedllamaCausalLM,
        abbr='pruned-llama-13b-hf-512red-2-38-layer_imp_weight-0.2',
        path="/home/zhangyihan/LMFlow/prune_log/layer_importance/llama-13b-mlp-mha-0-40-0.225-8192red1024-paramfirst-firstabs",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]

#full eval
datasets = [*humaneval_datasets, *mbpp_datasets, *hellaswag_datasets,  *piqa_datasets, *siqa_datasets, *winogrande_datasets, *ARC_e_datasets, *ARC_c_datasets, *obqa_datasets, *commonsenseqa_datasets, *nq_datasets, *nq_retrieval_datasets, *triviaqa_datasets, *triviaqa_retrieval_datasets, *squad20_datasets, *BoolQ_datasets, *gsm8k_datasets,  *math_datasets, *mmlu_datasets, *bbh_datasets]

# #short eval
# datasets = [*hellaswag_datasets,  *piqa_datasets, *siqa_datasets, *winogrande_datasets, *ARC_e_datasets, *ARC_c_datasets, *obqa_datasets, *commonsenseqa_datasets, *nq_datasets, *triviaqa_datasets, *squad20_datasets, *BoolQ_datasets, *mmlu_datasets]
