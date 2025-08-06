#!/bin/bash

parallel_num=4

data_dir="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"

dataset_names=(
    "MTBench"
    "AlpacaEval"
    "ArenaHard"
    # "MLRBench"
)

assign_model="gpt-4.1-mini"

counter=0

for i in ${!dataset_names[@]}; do
    python -m src.data.assign_category \
        --dataset_name ${dataset_names[i]} \
        --assign_model_name ${assign_model} \
        --data_dir ${data_dir} &
        
    counter=$((counter + 1))
    if [ $((counter % parallel_num)) -eq 0 ]; then
        wait
    fi
done
wait
