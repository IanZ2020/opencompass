from mmengine.config import read_base

with read_base():
    from .datasets.collections.base_medium_llama import humaneval_datasets, hellaswag_datasets,  piqa_datasets, siqa_datasets, winogrande_datasets, ARC_e_datasets, ARC_c_datasets, mbpp_datasets, obqa_datasets, commonsenseqa_datasets, nq_datasets, triviaqa_datasets, squad20_datasets, BoolQ_datasets, gsm8k_datasets, math_datasets
    from .models.hf_llama.hf_llama_13b import models


datasets = [*humaneval_datasets,*mbpp_datasets, *hellaswag_datasets, *piqa_datasets, *siqa_datasets, *winogrande_datasets, *ARC_e_datasets, *ARC_c_datasets,  *obqa_datasets, *commonsenseqa_datasets, *nq_datasets, *triviaqa_datasets, *squad20_datasets, *BoolQ_datasets, *gsm8k_datasets,  *math_datasets]

# 
