'''
Dataset for evaluation:
1. AlpacaEval
2. ArenaHard
3. MTBench
4. UltraFeedback
5. Dolly
6. Oasst
7. JudgeBench
'''
import json
import os
import asyncio

from src.logging_utils import setup_logging
from src.llm_zoo import load_model
from src.data.data_utils import load_metadata


def get_response_from_model(save_dir, dataset_name, response_model_name, **kwargs):
    '''
    Get the response from the model and save it to the local directory.
    Args:
        save_dir: str, the directory to save the dataset
        dataset_name: str, the name of the dataset
        response_model_name: str, the name of the model to get the response
        **kwargs: dict, the arguments to pass to the load_model function
    '''
    metadata = load_metadata(save_dir, dataset_name)
    questions = [item["instruction"] for item in metadata]

    response_model = load_model(response_model_name, **kwargs)
    if "gpt" in response_model_name:
        responses = asyncio.run(response_model.batch_invoke(questions))
    else:
        responses = response_model.batch_invoke(questions)
    
    new_dataset = []
    for item, response in zip(metadata, responses):
        item['output'] = response
        new_dataset.append(item.copy())
    
    if "/" in response_model_name:
        response_model_name = response_model_name.split("/")[-1]
    
    save_path = os.path.join(save_dir, dataset_name, f"{response_model_name}.json")
    with open(save_path, "w") as f:
        json.dump(new_dataset, f, indent=4)
    return new_dataset


if __name__ == "__main__":

    data_dir = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"

    # ------------------------------------------------------------
    # get response from model
    # ------------------------------------------------------------
    setup_logging(task_name="get_response_from_model")

    dataset_list = [
        "AlpacaEval",
        "ArenaHard",
        "MTBench",
    ]
    # vllm based models
    vllm_model_name_list = [
        # "meta-llama/Llama-3.2-1B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        # "google/gemma-3-1b-it",
        # "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "mistralai/Mistral-7B-Instruct-v0.2"
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ]

    for dataset_name in dataset_list:
        for model_name in vllm_model_name_list:
            get_response_from_model(data_dir, dataset_name, model_name, use_vllm=True, tensor_parallel_size=8, gpu_memory_utilization=0.8)

    # api based models
    api_model_name_list = [
        # "gpt-4o-mini",
        # "gpt-4o",
        # "gpt-4.1",
        "gpt-4.1-mini",
        # "gpt-4o-2024-05-13",
        # "gpt-4-turbo-2024-04-09",
        # "gpt-4-0613",
        # "gemini-1.5-pro",
        # "gemini-1.5-flash"
    ]

    for dataset_name in dataset_list:
        for model_name in api_model_name_list:
            get_response_from_model(data_dir, dataset_name, model_name)
    
