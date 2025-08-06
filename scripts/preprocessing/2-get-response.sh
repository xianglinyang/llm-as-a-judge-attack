#!/bin/bash

data_dir="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"

dataset_names=(
    "MTBench"
    "AlpacaEval"
    "ArenaHard"
    # "MLRBench"
)
response_models=(
    "gpt-4.1-mini"
)


for i in ${!dataset_names[@]}; do
    for j in ${!response_models[@]}; do
        python -m src.data.get_response_data \
            --dataset_name ${dataset_names[i]} \
            --response_model_name ${response_models[i]} \
            --judge_backbone_name ${judge_backbones[i]} \
            --data_dir ${data_dir} \
            # --use_vllm \
done
