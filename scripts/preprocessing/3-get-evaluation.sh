#!/bin/bash

parallel_num=4

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
judge_models=(
    "gemini-2.0-flash"
)

judge_type="pointwise"

counter=0

for i in ${!dataset_names[@]}; do
    for j in ${!response_models[@]}; do
        python -m src.data.get_evaluation_score \
            --dataset_name ${dataset_names[i]} \
            --response_model_name ${response_models[i]} \
            --judge_model_name ${judge_models[i]} \
            --judge_type ${judge_type} \
            --data_dir ${data_dir} &
        counter=$((counter + 1))
        if [ $((counter % parallel_num)) -eq 0 ]; then
            wait
        fi
    done
done
wait