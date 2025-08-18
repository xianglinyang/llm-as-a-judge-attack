#!/bin/bash

# Parallel num
# parallel_num=10

# GPU Configuration
per_gpu_jobs_num=1
gpu_num=1
jobs_num=$((per_gpu_jobs_num*gpu_num))
gpu_ids=(0 1 2 3 4 5 6 7)


# LLM Agents
llm_agents=(
    "gpt-4.1-nano"
)

# Response Models
response_models=(
    "gpt-4.1-mini"
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

eval_num=100


datasets=(
    "UltraFeedback"
)

reward_types=(
    # "relative"
    "absolute"
)

lambda_reg=1.0
alpha=1.2
burnin_passes=1
ucb_passes=20
epsilon=0.15
ci_width_threshold=0.12
patience=3


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

for dataset in ${datasets[@]}; do
    for judge_model_name in ${judge_model_names[@]}; do
        for judge_type in ${judge_types[@]}; do
            for llm_agent_name in ${llm_agents[@]}; do
                for response_model_name in ${response_models[@]}; do
                    for reward_type in ${reward_types[@]}; do

                        # parallel num
                        gpu_id=${gpu_ids[$((counter % jobs_num))]}
                        echo "Starting job $((counter + 1)): Dataset=${dataset}, Judge=${judge_model_name}, Type=${judge_type}, LLM=${llm_agent_name}, Response=${response_model_name}, Reward=${reward_type}"
                        
                        CUDA_VISIBLE_DEVICES=${gpu_id} python -m src.evolve_agent.bandit.init_LinUCB_warmup \
                        --dataset_name ${dataset} \
                        --judge_model_name ${judge_model_name} \
                        --judge_type ${judge_type} \
                        --llm_agent_name ${llm_agent_name} \
                        --response_model_name ${response_model_name} \
                        --reward_type ${reward_type} \
                        --lambda_reg ${lambda_reg} \
                        --alpha ${alpha} \
                        --burnin_passes ${burnin_passes} \
                        --ucb_passes ${ucb_passes} \
                        --epsilon ${epsilon} \
                        --ci_width_threshold ${ci_width_threshold} \
                        --patience ${patience} \
                        --eval_num ${eval_num} \
                        --save_model_path /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/models/ \
                        --save_analysis_path results/ \
                        --save_metrics_path /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/metrics/ &

                        # Increment counter
                        counter=$((counter + 1))    
                        echo "Launched job $counter"

                        if [ $((counter % jobs_num)) -eq 0 ]; then
                            echo "Waiting for batch of $jobs_num jobs to complete..."
                            wait
                        fi
                    done
                done
            done
        done
    done
done

wait
echo "All UCB warmup jobs completed! Total jobs launched: $counter"
