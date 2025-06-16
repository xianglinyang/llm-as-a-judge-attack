# llm-as-a-judge-attack

# Data Preprocess
1. download the dataset according to our format in ```preprocessing_data.ipynb```
2. get the category of each question by running (need to modify the file)
```console
python -m src.data
```
3. get the response of the target model by running (need to modify the file)

```console
python -m src.data
```

# Explore with Bandit Algorithm

```console
CUDA_VISIBLE_DEVICES=0 python -m src.evolve_agent.bandit.UCB --args....
```


# Notes
1. check api key rate limit
2. 



# Reference
Bandit algorithm adapt from https://github.com/xianglinyang/TS.git