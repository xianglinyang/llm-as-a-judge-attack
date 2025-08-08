#!/bin/bash

# Parallel num
parallel_num=2

# # GPU Configuration
# per_gpu_jobs_num=1
# gpu_num=8
# jobs_num=$((per_gpu_jobs_num*gpu_num))
# gpu_ids=(0 1 2 3 4 5 6 7)

# Hyperparameters
budgets=(20 30)
pool_sizes=(3 5)

# LLM Agents
llm_agents=(
    "gemini-2.5-flash-lite"
    "gpt-4.1-nano"
)

# Response Models
response_models=(
    "openai/gpt-4.1-mini"
    "gemini-1.5-flash-8b"
    "Qwen/Qwen3-14B"
)

# Judge Models
judge_model_names=(
    "google/gemini-2.5-flash"
    "openai/o3"
    "openai/o3-mini"
    "google/gemini-2.5-pro"
    "qwen/qwen3-235b-a22b-thinking-2507"
    "qwen/qwen3-235b-a22b-2507"
    "deepseek/deepseek-r1-0528"
    "anthropic/claude-sonnet-4"
    "meta-llama/llama-3.3-70b-instruct"
    "qwen/qwen3-30b-a3b-instruct-2507"
)

# Judge Types
judge_types=(
    "pointwise"
    # "pairwise_fine_grained"
    # "pairwise"
    # "mlr_bench"
)

eval_num=1000


datasets=(
    "MTBench"
    "AlpacaEval"
    "ArenaHard"
    # "MLRBench"
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
            for response_model in ${response_models[@]}; do
                for judge_model_name in ${judge_model_names[@]}; do
                    for judge_type in ${judge_types[@]}; do
                        for dataset in ${datasets[@]}; do
                            # parallel num
                            python -m src.evolve_agent.direct_prompting \
                            --judge_type ${judge_type} \
                            --Budget ${budget} \
                            --pool_size ${pool_size} \
                            --judge_backbone ${judge_model_name} \
                            --llm_agent_name ${llm_agent} \
                            --dataset_name ${dataset} \
                            --response_model_name ${response_model} \
                            --eval_num ${eval_num} \
                            --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data \
                            --save_trajectory_path /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/ &

                            # Increment counter
                            counter=$((counter + 1))
                            
                            # If we've launched jobs_num jobs, wait for them to complete
                            if [ $((counter % parallel_num)) -eq 0 ]; then
                                wait
                            fi
                        done
                    done
                done
            done
        done
    done
    done


wait

# ---- Pairwise ----
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
#                                             python -m src.evolve_agent.direct_prompting \
#                                             --judge_type ${judge_type} \
#                                             --Budget ${budget} \
#                                             --pool_size ${pool_size} \
#                                             --judge_backbone ${judge_model_name} \
#                                             --llm_agent_name ${llm_agent} \
#                                             --dataset_name ${dataset} \
#                                             --response_model_name ${response_model} \
#                                             --eval_num ${eval_num} \
#                                             --answer_position ${answer_position} \
#                                             --baseline_response_model_name ${baseline_response_model} \
#                                             --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data \
#                                             --save_trajectory_path /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/ &

#                                             # Increment counter
#                                             counter=$((counter + 1))

#                                             # If we've launched jobs_num jobs, wait for them to complete
#                                             if [ $((counter % parallel_num)) -eq 0 ]; then
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
#                             # parallel num
#                             python -m src.evolve_agent.direct_prompting \
#                             --judge_type ${judge_type} \
#                             --Budget ${budget} \
#                             --pool_size ${pool_size} \
#                             --judge_backbone ${judge_model_name} \
#                             --llm_agent_name ${llm_agent} \
#                             --dataset_name MLRBench \
#                             --response_model_name ${response_model} \
#                             --eval_num ${eval_num} \
#                             --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data \
#                             --save_trajectory_path /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/ &

#                             # Increment counter
#                             counter=$((counter + 1))
                            
#                             if [ $((counter % parallel_num)) -eq 0 ]; then  
#                                 wait
#                             fi
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done
# wait