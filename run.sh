# # dataset: AlpacaEval
# CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.baseline_prompt --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300

# CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.baseline_prompt --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300


# CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --test_mode random --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300

# CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --test_mode single --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300

# CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --test_mode online --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300

# CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --test_mode random --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode single --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode online --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300


# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode random --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode single --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode online --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode random --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode single --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode online --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300



# dataset: ArenaHard
# parallel run




# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.baseline_prompt --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.baseline_prompt --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &


# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode random --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode single --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode online --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode random --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode single --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode online --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &


# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode random --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode single --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode online --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode random --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode single --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

# CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode online --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &


CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --judge_type pairwise --answer_position first --baseline_response_model_name gpt-4o-2024-05-13 --test_mode single --Budget 30 --pool_size 3 --judge_model_name gemini-2.0-flash --llm_agent_name gpt-4.1-nano --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /data2/xianglin/llm-as-a-judge-attack/data --eval_num 805

CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --judge_type pairwise_fine_grained --answer_position first --baseline_response_model_name gpt-4o-2024-05-13 --test_mode single --Budget 30 --pool_size 3 --judge_model_name gemini-2.0-flash --llm_agent_name gpt-4.1-nano --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /data2/xianglin/llm-as-a-judge-attack/data --eval_num 805

CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --judge_type pairwise --answer_position first --baseline_response_model_name gpt-4o-2024-05-13 --test_mode random --Budget 30 --pool_size 3 --judge_model_name gemini-2.0-flash --llm_agent_name gpt-4.1-nano --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /data2/xianglin/llm-as-a-judge-attack/data --eval_num 805

CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.direct_prompting --judge_type pairwise --answer_position first --baseline_response_model_name gpt-4o-2024-05-13 --Budget 30 --pool_size 3 --judge_backbone gemini-2.0-flash --llm_agent_name gpt-4.1-nano --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /data2/xianglin/llm-as-a-judge-attack/data --eval_num 805


CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --reward_type relative --judge_type mlr_bench --test_mode single --Budget 30 --pool_size 3 --judge_model_name o4-mini --llm_agent_name gpt-4.1-nano --dataset_name MLRBench --response_model_name claude-3-7-sonnet-20250219 --data_dir /data2/xianglin/llm-as-a-judge-attack/data --eval_num 10 --save_trajectory_path /data2/xianglin/llm-as-a-judge-attack/trajectories/

CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --reward_type relative --judge_type mlr_bench --test_mode random --Budget 30 --pool_size 3 --judge_model_name o4-mini --llm_agent_name gpt-4.1-nano --dataset_name MLRBench --response_model_name claude-3-7-sonnet-20250219 --data_dir /data2/xianglin/llm-as-a-judge-attack/data --eval_num 10 --save_trajectory_path /data2/xianglin/llm-as-a-judge-attack/trajectories/

CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.direct_prompting --judge_type mlr_bench --Budget 30 --pool_size 3 --judge_backbone o4-mini --llm_agent_name gpt-4.1-nano --dataset_name MLRBench --response_model_name claude-3-7-sonnet-20250219 --data_dir /data2/xianglin/llm-as-a-judge-attack/data --eval_num 10 --save_trajectory_path /data2/xianglin/llm-as-a-judge-attack/trajectories/


CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --reward_type relative --judge_type mlr_bench --test_mode single --Budget 30 --pool_size 3 --judge_model_name o4-mini --llm_agent_name gpt-4.1-nano --dataset_name MLRBench --response_model_name o4-mini --data_dir /data2/xianglin/llm-as-a-judge-attack/data --eval_num 10 --save_trajectory_path /data2/xianglin/llm-as-a-judge-attack/trajectories/

CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --reward_type relative --judge_type mlr_bench --test_mode random --Budget 30 --pool_size 3 --judge_model_name o4-mini --llm_agent_name gpt-4.1-nano --dataset_name MLRBench --response_model_name o4-mini --data_dir /data2/xianglin/llm-as-a-judge-attack/data --eval_num 10 --save_trajectory_path /data2/xianglin/llm-as-a-judge-attack/trajectories/

CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.direct_prompting --judge_type mlr_bench --Budget 30 --pool_size 3 --judge_backbone o4-mini --llm_agent_name gpt-4.1-nano --dataset_name MLRBench --response_model_name o4-mini --data_dir /data2/xianglin/llm-as-a-judge-attack/data --eval_num 10 --save_trajectory_path /data2/xianglin/llm-as-a-judge-attack/trajectories/



CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --reward_type relative --judge_type pointwise --test_mode single --Budget 20 --pool_size 3 --judge_model_name gemini-2.0-flash --llm_agent_name gpt-4.1-nano --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /data2/xianglin/llm-as-a-judge-attack/data --save_trajectory_path /data2/xianglin/llm-as-a-judge-attack/trajectories --eval_num 805







