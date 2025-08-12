#!/bin/bash

parallel_num=1

# GPU Configuration
per_gpu_jobs_num=1
gpu_num=8
jobs_num=$((per_gpu_jobs_num*gpu_num))
gpu_ids=(0 1 2 3 4 5 6 7)

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

# ------------------------------------------------------------------------------------------------
# api call
# ------------------------------------------------------------------------------------------------

counter=0

for i in ${!dataset_names[@]}; do
    for j in ${!response_models[@]}; do
        python -m src.data.get_response_data \
            --dataset_name ${dataset_names[i]} \
            --response_model_name ${response_models[j]} \
            --data_dir ${data_dir} &
            
        counter=$((counter + 1))
        if [ $((counter % parallel_num)) -eq 0 ]; then
            wait
        fi
    done
done
wait

# ------------------------------------------------------------------------------------------------
# vllm call
# ------------------------------------------------------------------------------------------------

# counter=0

# for i in ${!dataset_names[@]}; do
#     for j in ${!response_models[@]}; do

#         CUDA_VISIBLE_DEVICES=${gpu_ids[$((counter % jobs_num))]} python -m src.data.get_response_data \
#             --dataset_name ${dataset_names[i]} \
#             --response_model_name ${response_models[j]} \
#             --use_vllm \
#             --tensor_parallel_size 1 \
#             --gpu_memory_utilization 0.95 \
#             --data_dir ${data_dir} &
            
#         counter=$((counter + 1))
#         if [ $((counter % parallel_num)) -eq 0 ]; then
#             wait
#         fi
#     done
# done
# wait
