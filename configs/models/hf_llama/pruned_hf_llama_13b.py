from opencompass.models import PrunedllamaCausalLM

models = [
    # LLaMA 13B
    dict(
        type=PrunedllamaCausalLM,
        abbr='pruned-llama-13b-hf-recover-red',
        path="/home/zhangyihan/LMFlow/prune_log/llama-13b-mlp-mha-2-38-0.25-4096red1024-paramfirst-firstabs",
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
