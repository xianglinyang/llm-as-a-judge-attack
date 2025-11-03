#!/bin/bash

parallel_num=1

data_dir="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"

dataset_names=(
    # "MTBench"
    # "AlpacaEval"
    # "ArenaHard"
    "MLRBench"
    # "UltraFeedback"
)
response_models=(
    # "gpt-4.1-mini"
    # "openai/gpt-4o"
    "gemini-2.5-pro-preview-03-25"
    "claude-3-7-sonnet-20250219"
    "o4-mini"
)

# Judge Models
judge_models=(
    "qwen/qwen3-235b-a22b-2507"
    "meta-llama/llama-3.3-70b-instruct"
    "deepseek/deepseek-r1-0528"
    "openai/o3-mini"
    "google/gemini-2.5-flash"
    # "openai/gpt-5"
    # "google/gemini-2.5-pro"
    # "anthropic/claude-opus-4"
    # "x-ai/grok-4"
    # "qwen/qwen3-max"
    # "openai/o4-mini"
)


judge_type="mlr_bench"

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