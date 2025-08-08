# llm-as-a-judge-attack

# Data Preprocess
1. download the dataset according to our format in ```download.ipynb``` under data dir
2. get the category of each question by running (need to modify the file)
```console
bash scripts/preprocessing/1-assign-categories.sh
```
3. get the response of the target model by running (need to modify the file)

```console
bash scripts/preprocessing/2-get-response.sh
```
4. get the evaluation of the response by running 
```console
bash scripts/preprocessing/3-get-evaluation.sh
```

# Explore with Bandit Algorithm
Run single evaluation
```console
CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.direct_prompting --judge_type pairwise --answer_position first --baseline_response_model_name gpt-4o-2024-05-13 --Budget 30 --pool_size 3 --judge_backbone gemini-2.0-flash --llm_agent_name gpt-4.1-nano --dataset_name AlpacaEval --response_model_name gpt-4o-mini --data_dir /path/to/llm-as-a-judge-attack/data --eval_num 805 --save_trajectory_path /path/to/llm-as-a-judge-attack/trajectories/

CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --args....
```
Scripts
```console
bash scripts/main/run_bias.sh
```

# Explore with Baselines
Scripts
```console
bash scripts/main/run_direct_prompting.sh
```

# Analysis
1. feature importance: regression.py
2. detection: explanation.ipynb
3. transferbility: transferability.ipynb

# Notes
1. check api key rate limit
2. put correct key in the env


# Reference
Bandit algorithm adapt from https://github.com/xianglinyang/TS.git