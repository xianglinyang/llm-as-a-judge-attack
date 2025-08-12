#!/bin/bash

parallel_num=1

data_dir="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"

dataset_names=(
    "MTBench"
    "AlpacaEval"
    "ArenaHard"
    # "MLRBench"
)
response_models=(
    "gpt-4.1-mini"
    # "gemini-1.5-flash-8b"
    # "Qwen/Qwen3-14B"
)

# Judge Models
judge_models=(
    "qwen/qwen3-235b-a22b-2507"
    "google/gemini-2.5-flash"
    "openai/gpt-5"
    "meta-llama/llama-3.3-70b-instruct"
    "deepseek/deepseek-r1-0528"
)


judge_type="pointwise"

counter=0

for i in ${!dataset_names[@]}; do
    for j in ${!response_models[@]}; do
        for k in ${!judge_models[@]}; do
            python -m src.data.get_evaluation_score \
                --dataset_name ${dataset_names[i]} \
                --response_model_name ${response_models[j]} \
                --judge_model_name ${judge_models[k]} \
                --judge_type ${judge_type} \
                --data_dir ${data_dir} &
            counter=$((counter + 1))
            if [ $((counter % parallel_num)) -eq 0 ]; then
                wait
            fi
        done
    done
done
wait