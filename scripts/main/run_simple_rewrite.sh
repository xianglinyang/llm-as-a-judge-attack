#!/bin/bash

# Parallel num
parallel_num=1

# # GPU Configuration
# per_gpu_jobs_num=1
# gpu_num=8
# jobs_num=$((per_gpu_jobs_num*gpu_num))
# gpu_ids=(0 1 2 3 4 5 6 7)

# Hyperparameters
budgets=(30)
pool_sizes=(3)

# LLM Agents
llm_agents=(
    # "gpt-5-nano"
    # "openai/gpt-4.1-nano"
    "gemini-1.5-flash-8b"
)

# Response Models
response_models=(
    # "gpt-4.1-mini"
    "claude-3-7-sonnet-20250219"
    "gemini-2.5-pro-preview-03-25"
    "o4-mini"
)

# Judge Models
judge_model_names=(
    "qwen/qwen3-235b-a22b-2507"
    "meta-llama/llama-3.3-70b-instruct"
    "deepseek/deepseek-r1-0528"
    "openai/o3-mini"
    # "gpt-5"
    "google/gemini-2.5-flash"
    
)

template_names=(
    # "holistic"
    "improve"
)

# Judge Types
judge_types=(
    # "pointwise"
    # "pairwise_fine_grained"
    # "pairwise"
    # "alpaca_eval"
    # "arena_hard_auto"
    "mlr_bench"
)

eval_num=10


datasets=(
    # "MTBench"
    # "AlpacaEval"
    # "ArenaHard"
    "MLRBench"
)


# # ---- Pointwise ----

# counter=0

# for budget in ${budgets[@]}; do
#     for pool_size in ${pool_sizes[@]}; do
#         for llm_agent in ${llm_agents[@]}; do
#             for response_model in ${response_models[@]}; do
#                 for judge_model_name in ${judge_model_names[@]}; do
#                     for judge_type in ${judge_types[@]}; do
#                         for dataset in ${datasets[@]}; do
#                         for template_name in ${template_names[@]}; do
#                                 # parallel num
#                                 python -m src.evolve_agent.simple_rewrite \
#                                 --judge_type ${judge_type} \
#                                 --Budget ${budget} \
#                                 --pool_size ${pool_size} \
#                                 --judge_backbone ${judge_model_name} \
#                                 --llm_agent_name ${llm_agent} \
#                                 --dataset_name ${dataset} \
#                                 --response_model_name ${response_model} \
#                                 --eval_num ${eval_num} \
#                                 --template_name ${template_name} \
#                                 --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data \
#                                 --save_trajectory_path /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/ &

#                                 # Increment counter
#                                 counter=$((counter + 1))
                                
#                                 # If we've launched jobs_num jobs, wait for them to complete
#                                 if [ $((counter % parallel_num)) -eq 0 ]; then
#                                     wait
#                                 fi
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
#     done


# wait

# # ---- Pairwise ----

# baseline_response_models=(
#     "gpt-4o"
# )

# answer_positions=(
#     "first"
#     "second"
# )
# counter=0

# for budget in ${budgets[@]}; do
#     for pool_size in ${pool_sizes[@]}; do
#         for llm_agent in ${llm_agents[@]}; do
#             for response_model in ${response_models[@]}; do
#                 for judge_model_name in ${judge_model_names[@]}; do
#                     for judge_type in ${judge_types[@]}; do
#                         for dataset in ${datasets[@]}; do
#                             for template_name in ${template_names[@]}; do
#                                 for answer_position in ${answer_positions[@]}; do
#                                     for baseline_response_model in ${baseline_response_models[@]}; do
#                                         # parallel num
#                                         python -m src.evolve_agent.simple_rewrite \
#                                         --judge_type ${judge_type} \
#                                         --Budget ${budget} \
#                                         --pool_size ${pool_size} \
#                                         --judge_backbone ${judge_model_name} \
#                                         --llm_agent_name ${llm_agent} \
#                                         --dataset_name ${dataset} \
#                                         --response_model_name ${response_model} \
#                                         --eval_num ${eval_num} \
#                                         --template_name ${template_name} \
#                                         --answer_position ${answer_position} \
#                                         --baseline_response_model_name ${baseline_response_model} \
#                                         --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data \
#                                         --save_trajectory_path /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/ &

#                                         # Increment counter
#                                         counter=$((counter + 1))
                                        
#                                         # If we've launched jobs_num jobs, wait for them to complete
#                                         if [ $((counter % parallel_num)) -eq 0 ]; then
#                                             wait
#                                         fi
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

# ---- MLRBench ----
counter=0

for budget in ${budgets[@]}; do
    for pool_size in ${pool_sizes[@]}; do
        for llm_agent in ${llm_agents[@]}; do
            for response_model in ${response_models[@]}; do
                for judge_model_name in ${judge_model_names[@]}; do
                    for judge_type in ${judge_types[@]}; do
                        # parallel num
                        python -m src.evolve_agent.simple_rewrite \
                        --judge_type ${judge_type} \
                        --Budget ${budget} \
                        --pool_size ${pool_size} \
                        --judge_backbone ${judge_model_name} \
                        --llm_agent_name ${llm_agent} \
                        --dataset_name MLRBench \
                        --response_model_name ${response_model} \
                        --eval_num ${eval_num} \
                        --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data \
                        --save_trajectory_path /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/ &

                        # Increment counter
                        counter=$((counter + 1))
                        
                        if [ $((counter % parallel_num)) -eq 0 ]; then  
                            wait
                        fi
                    done
                done
            done
        done
    done
done

wait