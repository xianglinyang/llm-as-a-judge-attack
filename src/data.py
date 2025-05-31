'''
1. AlpacaEval
2. ArenaHard
3. MTBench
4. UltraFeedback
5. Dolly
6. Oasst
'''
# parse the data_path.yaml
import json
import os
from tqdm import tqdm
import asyncio

from src.llm_zoo import VLLMModel, OpenAIModel
from src.utils import str2json
from src.logging_utils import setup_logging
from src.llm_zoo import load_model

def download_dataset(dataset_name, save_dir):
    '''
    Download the dataset from the internet and save it to the local directory.
    save it with format:
    [
        {
            "instruction": str,
            "other metadata": str,
        },
        ...
    ]
    Args:
        dataset_name: str, the name of the dataset
        save_dir: str, the directory to save the dataset
    '''
    if dataset_name == "AlpacaEval":
        pass
    elif dataset_name == "ArenaHard":
        pass
    elif dataset_name == "MTBench":
        pass
    elif dataset_name == "UltraFeedback":
        pass
    elif dataset_name == "Dolly":
        pass
    elif dataset_name == "Oasst":
        pass
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

def get_response_from_model(save_dir, dataset_name, response_model_name, **kwargs):
    '''
    Get the response from the model and save it to the local directory.
    Args:
        save_dir: str, the directory to save the dataset
        dataset_name: str, the name of the dataset
        response_model_name: str, the name of the model to get the response
        **kwargs: dict, the arguments to pass to the load_model function
    '''
    save_path = os.path.join(save_dir, dataset_name, f"metadata.json")
    assert os.path.exists(save_path), f"Save path {save_path} does not exist. Please download the dataset first."
    with open(save_path, "r") as f:
        dataset = json.load(f)
    
    questions = [item["instruction"] for item in dataset]

    response_model = load_model(response_model_name, **kwargs)
    if "gpt" in response_model_name:
        responses = asyncio.run(response_model.batch_invoke(questions))
    else:
        responses = response_model.batch_invoke(questions)
    
    new_dataset = []
    for item, response in zip(dataset, responses):
        item['output'] = response
        new_dataset.append(item.copy())
    
    if "/" in response_model_name:
        response_model_name = response_model_name.split("/")[-1]
    
    save_path = os.path.join(save_dir, dataset_name, f"{response_model_name}.json")
    with open(save_path, "w") as f:
        json.dump(new_dataset, f, indent=4)
    return new_dataset

def load_dataset(save_dir, dataset_name, response_model_name):
    metadata_path = os.path.join(save_dir, dataset_name, f"metadata.json")
    save_path = os.path.join(save_dir, dataset_name, f"{response_model_name}.json")
    if not os.path.exists(save_path):
        raise ValueError(f"Dataset {dataset_name} with response model {response_model_name} not found")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except:
        raise ValueError(f"Metadata {metadata_path} not found")
    
    try:
        with open(save_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Dataset {dataset_name} with response model {response_model_name} not found")
    
    # merge the metadata and data based on instruction field
    # TODO: optimize the merge process with dict structure
    new_dataset = []
    for item in metadata:
        for data_item in data:
            if item["instruction"] == data_item["instruction"]:
                new_item = item.copy()
                new_item.update(data_item)
                new_dataset.append(new_item)
                break
        else:
            new_dataset.append(item)
    return new_dataset

template='''Given a question, please categorize it to one of the following categories:

1. Computer Science & Programming
2. Mathematics & Statistics
3. Science & Engineering
4. Business & Finance
5. Writing & Communication
6. Social & Daily Life
7. Others

## Question: {}

Please output the generated content in a json format, for example:
```json
{{
"category": "integer from 1 to 7" // specific category, such as 1 for "Computer Science & Programming"
}}
```

Formatted the abovementioned schema and categorize the given question:'''

CATEGORIES = [
    "Computer Science & Programming",
    "Mathematics & Statistics",
    "Science & Engineering",
    "Business & Finance",
    "Writing & Communication",
    "Social & Daily Life",
    "Others"
]

def assign_category(save_dir, dataset_name):
    save_path = os.path.join(save_dir, dataset_name, f"metadata.json")
    assert os.path.exists(save_path), f"Save path {save_path} does not exist. Please download the dataset first."
    with open(save_path, "r") as f:
        dataset = json.load(f)
    
    # check if the dataset is already assigned category
    if "category" in dataset[0]:
        return dataset
    
    model = OpenAIModel("gpt-4o-mini")
    for item in tqdm(dataset):
        prompt = template.format(item["instruction"])
        response = model.invoke(prompt)
        try:
            category = int(str2json(response)["category"])
            item["category"] = CATEGORIES[category - 1]
        except:
            item["category"] = "None"
    
    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=4)
    return dataset

if __name__ == "__main__":
    save_dir = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"

    
    # ------------------------------------------------------------
    # assign category
    # setup_logging(task_name="assign_category")
    
    # dataset_name = "AlpacaEval"
    # assign_category(save_dir, dataset_name)

    # dataset_name = "ArenaHard"
    # assign_category(save_dir, dataset_name)

    # dataset_name = "MTBench"
    # assign_category(save_dir, dataset_name)

    # ------------------------------------------------------------
    # get response from model
    setup_logging(task_name="get_response_from_model")

    model_name_list = [
        # "meta-llama/Llama-3.2-1B-Instruct",
        # "meta-llama/Llama-3.2-3B-Instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "google/gemma-3-1b-it",
        # "google/gemma-3-4b-it",
        # "google/gemma-3-12b-it",
        # "google/gemma-3-27b-it",
        # "mistralai/Mixtral-7B-Instruct-v0.2",
        # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ]

    # dataset_name = "AlpacaEval"
    # for model_name in model_name_list:
    #     get_response_from_model(save_dir, dataset_name, model_name, use_vllm=True, tensor_parallel_size=8)

    # dataset_name = "ArenaHard"
    # for model_name in model_name_list:
    #     get_response_from_model(save_dir, dataset_name, model_name, use_vllm=True, tensor_parallel_size=8)

    # dataset_name = "MTBench"
    # for model_name in model_name_list:
    #     get_response_from_model(save_dir, dataset_name, model_name, use_vllm=True, tensor_parallel_size=8)

    model_name_list = [
        "gpt-4o-mini",
        # "gpt-4o",
        # "gpt-4.1",
        # "gpt-4.1-mini",
        # "gpt-4o-2024-05-13",
        # "gpt-4-turbo-2024-04-09",
        # "gpt-4-0613",
        # "gemini-1.5-pro",
        # "gemini-1.5-flash"
    ]

    dataset_name = "AlpacaEval"
    for model_name in model_name_list:
        get_response_from_model(save_dir, dataset_name, model_name)

    dataset_name = "ArenaHard"
    for model_name in model_name_list:
        get_response_from_model(save_dir, dataset_name, model_name)

    dataset_name = "MTBench"
    for model_name in model_name_list:
        get_response_from_model(save_dir, dataset_name, model_name)
    
    
    
    
