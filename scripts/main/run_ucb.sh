#!/bin/bash

# Parallel num
# parallel_num=10

# GPU Configuration
per_gpu_jobs_num=1
gpu_num=2
jobs_num=$((per_gpu_jobs_num*gpu_num))
gpu_ids=(0 1 2 3 4 5 6 7)

# Hyperparameters
budgets=(20)
pool_sizes=(3)

# LLM Agents
llm_agents=(
    # "gemini-2.5-flash-lite"
    "gpt-4.1-nano"
    # "gemini-1.5-flash-8b"
)

# Response Models
response_models=(
    "gpt-4.1-mini"
    # "gemini-1.5-flash-8b"
    # "Qwen/Qwen3-14B"
)

# Judge Models
judge_model_names=(
    "qwen/qwen3-235b-a22b-2507"
    # "qwen/qwen3-30b-a3b-instruct-2507"
    # "meta-llama/llama-3.3-70b-instruct"
    # "deepseek/deepseek-r1-0528"
    # "gemini-2.0-flash"
    # "gemini-2.5-flash"
)

# Judge Types
judge_types=(
    "pointwise"
    # "pairwise_fine_grained"
    # "pairwise"
    # "mlr_bench"
)

eval_num=5

test_modes=(
    "ucb"
    "random"
)

datasets=(
    # "MTBench"
    "AlpacaEval"
    # "ArenaHard"
    # "MLRBench"
)

reward_types=(
    "relative"
    # "absolute"
)

cold_start="0"

init_model_paths=(
    "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/models/warmup_models/judge_qwen3-235b-a22b-2507/response_gpt-4.1-mini/baseline_None/reward_relative/alpha_1.2_lambda_1.0/ci_0.12_patience_2/20250818_163006"
)

# ---- pairwise ----

# baseline_response_models=(
#     "gpt-4o-2024-05-13"
#     "gpt-4o-2024-05-13"
# )

# answer_positions=(
#     "first"
#     "second"
# )



# ---- Pointwise ----
counter=0

for budget in ${budgets[@]}; do
    for pool_size in ${pool_sizes[@]}; do
        for llm_agent in ${llm_agents[@]}; do
            for test_mode in ${test_modes[@]}; do
                for response_model in ${response_models[@]}; do
                    for judge_type in ${judge_types[@]}; do
                        for dataset in ${datasets[@]}; do
                            for reward_type in ${reward_types[@]}; do
                                # zip judge_model_name and init_model_path
                                for (( i=0; i<${#judge_model_names[*]}; ++i)); do
                                    judge_model_name=${judge_model_names[$i]}
                                    init_model_path=${init_model_paths[$i]}

                                    # parallel num
                                    gpu_id=${gpu_ids[$((counter % jobs_num))]}


                                    CUDA_VISIBLE_DEVICES=${gpu_id} python -m src.evolve_agent.bandit.UCB \
                                    --judge_type ${judge_type} \
                                    --test_mode ${test_mode} \
                                    --Budget ${budget} \
                                    --pool_size ${pool_size} \
                                    --judge_model_name ${judge_model_name} \
                                    --llm_agent_name ${llm_agent} \
                                    --dataset_name ${dataset} \
                                    --response_model_name ${response_model} \
                                    --eval_num ${eval_num} \
                                    --cold_start ${cold_start} \
                                    --init_model_path ${init_model_path} \
                                    --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data \
                                    --save_trajectory_path /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/ &

                                    # Increment counter
                                    counter=$((counter + 1))
                                    
                                    # If we've launched jobs_num jobs, wait for them to complete
                                    if [ $((counter % jobs_num)) -eq 0 ]; then
                                        wait
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

wait

# # ---- Pairwise ----
# counter=0

# for test_mode in ${test_modes[@]}; do
#     for budget in ${budgets[@]}; do
#         for pool_size in ${pool_sizes[@]}; do
#             for llm_agent in ${llm_agents[@]}; do
#                 for response_model in ${response_models[@]}; do
#                     for judge_model_name in ${judge_model_names[@]}; do
#                         for judge_type in ${judge_types[@]}; do
#                             for dataset in ${datasets[@]}; do
#                                 for answer_position in ${answer_positions[@]}; do
#                                     for baseline_response_model in ${baseline_response_models[@]}; do
#                                         for reward_type in ${reward_types[@]}; do
#                                             # parallel num
#                                             python -m src.evolve_agent.bandit.UCB \
#                                             --judge_type ${judge_type} \
#                                             --test_mode ${test_mode} \
#                                             --Budget ${budget} \
#                                             --pool_size ${pool_size} \
#                                             --judge_model_name ${judge_model_name} \
#                                             --llm_agent_name ${llm_agent} \
#                                             --dataset_name ${dataset} \
#                                             --response_model_name ${response_model} \
#                                             --eval_num ${eval_num} \
#                                             --answer_position ${answer_position} \
#                                             --baseline_response_model ${baseline_response_model} \
#                                             --reward_type ${reward_type} \
#                                             --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data \
#                                             --save_trajectory_path /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/ &

#                                             # Increment counter
#                                             counter=$((counter + 1))

#                                             # If we've launched jobs_num jobs, wait for them to complete
#                                             if [ $((counter % jobs_num)) -eq 0 ]; then
#                                                 wait
#                                             fi
#                                         done
#                                     done
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done
# wait

# # ---- MLRBench ----
# counter=0

# for test_mode in ${test_modes[@]}; do
#     for budget in ${budgets[@]}; do
#         for pool_size in ${pool_sizes[@]}; do
#             for llm_agent in ${llm_agents[@]}; do
#                 for response_model in ${response_models[@]}; do
#                     for judge_model_name in ${judge_model_names[@]}; do
#                         for judge_type in ${judge_types[@]}; do
#                             for dataset in ${datasets[@]}; do
#                                 for reward_type in ${reward_types[@]}; do
#                                     # parallel num
#                                     python -m src.evolve_agent.bandit.UCB \
#                                     --judge_type ${judge_type} \
#                                     --test_mode ${test_mode} \
#                                     --Budget ${budget} \
#                                     --pool_size ${pool_size} \
#                                     --judge_model_name ${judge_model_name} \
#                                     --llm_agent_name ${llm_agent} \
#                                     --dataset_name ${dataset} \
#                                     --response_model_name ${response_model} \
#                                     --eval_num ${eval_num} \
#                                     --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data \
#                                     --save_trajectory_path /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/ &

#                                     # Increment counter
#                                     counter=$((counter + 1))
                                    
#                                     if [ $((counter % jobs_num)) -eq 0 ]; then  
#                                         wait
#                                     fi
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done
# wait