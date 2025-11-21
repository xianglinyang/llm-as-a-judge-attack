#!/bin/bash

# Parallel num
# parallel_num=10

# GPU Configuration
per_gpu_jobs_num=1
gpu_num=1
jobs_num=$((per_gpu_jobs_num*gpu_num))
gpu_ids=(0 1)

# Hyperparameters
budgets=(25)
pool_sizes=(3)

# LLM Agents
llm_agents=(
    # "gpt-4.1-nano"
    # "gemini-1.5-flash-8b"
    # "openai/gpt-4.1-nano"
    "gemini-2.0-flash-lite"
)

# Response Models
response_models=(
    # "gpt-4.1-mini"
    "gpt-4o"
    # "claude-3-7-sonnet-20250219"
    # "gemini-2.5-pro-preview-03-25"
    # "o4-mini"
)

# Judge Models
judge_model_names=(
    # "qwen/qwen3-235b-a22b-2507"
    # "meta-llama/llama-3.3-70b-instruct"
    # "deepseek/deepseek-r1-0528"
    # "google/gemini-2.5-flash"
    # "openai/o3-mini"
    # "openai/gpt-5"
    # "openai/o4-mini"
    "deepseek/deepseek-r1-0528-qwen3-8b"
    "qwen/qwen3-8b"
    "meta-llama/llama-3-8b-instruct"
)

init_model_paths=(
    # For UCB with warmup, specify the path to pre-trained model:
    # /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/models/warmup_models/judge_qwen3-235b-a22b-2507/response_gpt-4.1-mini/baseline_None/reward_absolute/alpha_1.2_lambda_1.0/ci_0.12_patience_3/20250818_233149"
    # "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/models/warmup_models/judge_qwen3-235b-a22b-2507/response_gpt-4.1-mini/baseline_None/reward_relative/alpha_1.2_lambda_1.0/ci_0.12_patience_3/20250819_154019"
    # "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/models/warmup_models/judge_llama-3.3-70b-instruct/response_gpt-4.1-mini/baseline_None/reward_absolute/alpha_1.2_lambda_1.0/ci_0.12_patience_3/20250819_172352"
    # "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/models/warmup_models/judge_llama-3.3-70b-instruct/response_gpt-4.1-mini/baseline_None/reward_relative/alpha_1.2_lambda_1.0/ci_0.12_patience_3/20250819_162334"

    # For standard UCB or random, this is ignored:
    None
    None
    None
)

# Judge Types
judge_types=(
    # "pointwise"
    # "pairwise_fine_grained"
    # "pairwise"
    # "mlr_bench"
    "alpaca_eval"
    # "arena_hard_auto"
    # "pointwise_randomized"
    # "mt_bench"
    # "pointwise_ignore_bias"
)

eval_num=805

test_modes=(
    "ucb"                      # UCB (standard, cold start)
    # "ucb_with_warmup"        # UCB with warmup (uses pre-trained model)
    # "random"                 # Random arm selection baseline
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




# ---- pairwise ----

baseline_response_models=(
    "gpt-4o"
)

answer_positions=(
    "first"
    # "second"
)



# # ---- Pointwise ----
# counter=0

# for budget in ${budgets[@]}; do
#     for pool_size in ${pool_sizes[@]}; do
#         for llm_agent in ${llm_agents[@]}; do
#             for test_mode in ${test_modes[@]}; do
#                 for response_model in ${response_models[@]}; do
#                     for judge_type in ${judge_types[@]}; do
#                         for dataset in ${datasets[@]}; do
#                             for reward_type in ${reward_types[@]}; do
#                                 # zip judge_model_name and init_model_path
#                                 for (( i=0; i<${#judge_model_names[*]}; ++i)); do
#                                     judge_model_name=${judge_model_names[$i]}
#                                     init_model_path=${init_model_paths[$i]}

#                                     # parallel num
#                                     gpu_id=${gpu_ids[$((counter % jobs_num))]}


#                                     # Set init_model_path based on test_mode:
#                                     # - UCB: Standard UCB (cold start, no init_model_path)
#                                     # - UCB with warmup: Use warmup model if available
#                                     # - Random: No init_model_path needed
#                                     if [ "${test_mode}" = "ucb_with_warmup" ]; then
#                                         init_model_arg="--init_model_path ${init_model_path}"
#                                     else
#                                         init_model_arg=""
#                                     fi

#                                     CUDA_VISIBLE_DEVICES=${gpu_id} python -m src.evolve_agent.bandit.UCB \
#                                     --judge_type ${judge_type} \
#                                     --test_mode ${test_mode} \
#                                     --Budget ${budget} \
#                                     --reward_type ${reward_type} \
#                                     --pool_size ${pool_size} \
#                                     --judge_model_name ${judge_model_name} \
#                                     --llm_agent_name ${llm_agent} \
#                                     --dataset_name ${dataset} \
#                                     --response_model_name ${response_model} \
#                                     --eval_num ${eval_num} \
#                                     ${init_model_arg} \
#                                     --data_dir /data2/xianglin/A40/llm-as-a-judge-attack/data \
#                                     --save_trajectory_path /data2/xianglin/A40/llm-as-a-judge-attack/control_traj/ \
#                                     --save_metrics_path /data2/xianglin/A40/llm-as-a-judge-attack/control_metric/ &

#                                     # Increment counter
#                                     counter=$((counter + 1))
                                    
#                                     # If we've launched jobs_num jobs, wait for them to complete
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

# ---- Pairwise ----
counter=0

for test_mode in ${test_modes[@]}; do
    for budget in ${budgets[@]}; do
        for pool_size in ${pool_sizes[@]}; do
            for llm_agent in ${llm_agents[@]}; do
                for response_model in ${response_models[@]}; do
                    for judge_model_name in ${judge_model_names[@]}; do
                        for judge_type in ${judge_types[@]}; do
                            for dataset in ${datasets[@]}; do
                                for answer_position in ${answer_positions[@]}; do
                                    for baseline_response_model in ${baseline_response_models[@]}; do
                                        for reward_type in ${reward_types[@]}; do
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
                                            --answer_position ${answer_position} \
                                            --baseline_response_model ${baseline_response_model} \
                                            --reward_type ${reward_type} \
                                            --data_dir /data2/xianglin/A40/llm-as-a-judge-attack/data \
                                            --save_trajectory_path /data2/xianglin/A40/llm-as-a-judge-attack/trajectories/ \
                                            --save_metrics_path /data2/xianglin/A40/llm-as-a-judge-attack/metrics/ &

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
    done
done
wait

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
#                                     --data_dir /data2/xianglin/A40/llm-as-a-judge-attack/data \
#                                     --save_trajectory_path /data2/xianglin/A40/llm-as-a-judge-attack/trajectories/ 
#                                     --save_metrics_path /data2/xianglin/A40/llm-as-a-judge-attack/metrics/ &

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