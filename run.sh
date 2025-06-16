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




CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.baseline_prompt --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.baseline_prompt --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &


CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode random --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode single --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode online --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode random --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode single --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.UCB --test_mode online --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &


CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode random --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode single --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode online --Budget 20 --pool_size 3 --judge_model_name gpt-4o --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode random --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode single --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &

CUDA_VISIBLE_DEVICES=1 python -m src.evolve_agent.bandit.TS --test_mode online --Budget 20 --pool_size 3 --judge_model_name gemini-1.5-flash --llm_agent_name gpt-4o-mini --dataset_name ArenaHard --response_model_name gpt-4o-mini --data_dir /mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data --eval_num 300 &