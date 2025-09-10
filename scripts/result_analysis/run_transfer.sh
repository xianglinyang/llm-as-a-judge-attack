#!/bin/bash


models_list=(
    "qwen/qwen3-235b-a22b-2507"
    "meta-llama/llama-3.3-70b-instruct"
    "deepseek/deepseek-r1-0528"
    "google/gemini-2.5-flash"
    "openai/o3-mini"
)

strategies_list=(
    "ucb"
    "simple_rewrite_improve"
    "random"
)

datasets_list=(
    "AlpacaEval"
    "ArenaHard"
)

judge_types_list=(
    "pointwise"
    "alpaca_eval"
    "arena_hard_auto"
)

answer_positions_list=(
    "first"
    "second"
)

baseline_model_names_list=(
    "gpt-4o"
)
# pointwise transfer analysis

for source_judge in ${models_list[@]}; do
    for target_judge in ${models_list[@]}; do
        for strategy in ${strategies_list[@]}; do
            for dataset in ${datasets_list[@]}; do
                if [ $source_judge == $target_judge ]; then
                    continue
                fi
                python -m src.results_analysis.transfer_analysis --source_judge $source_judge --target_judge $target_judge --strategy $strategy --dataset_name $dataset --judge_type pointwise
            done
        done
    done
done

# # pairwise transfer analysis
# for source_judge in ${models_list[@]}; do
#     for target_judge in ${models_list[@]}; do
#         for strategy in ${strategies_list[@]}; do
#             for dataset in ${datasets_list[@]}; do
#                 for judge_type in ${judge_types_list[@]}; do
#                     for answer_position in ${answer_positions_list[@]}; do
#                         if [ $source_judge == $target_judge ]; then
#                             continue
#                         fi
#                         python -m src.results_analysis.transfer_analysis --source_judge $source_judge --target_judge $target_judge --strategy $strategy --dataset_name $dataset --judge_type $judge_type --answer_position $answer_position --baseline_model_name $baseline_model_name
#                     done
#                 done
#             done
#         done
#     done
# done
