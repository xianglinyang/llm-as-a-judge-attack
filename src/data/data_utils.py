import os
import json
import logging

logger = logging.getLogger(__name__)

def load_metadata(save_dir, dataset_name):
    metadata_path = os.path.join(save_dir, dataset_name, f"metadata.json")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except:
        raise ValueError(f"Metadata {metadata_path} not found")
    
    return metadata

def load_response(save_dir, dataset_name, response_model_name):
    save_path = os.path.join(save_dir, dataset_name, f"{response_model_name}.json")
    if not os.path.exists(save_path):
        raise ValueError(f"Dataset {dataset_name} with response model {response_model_name} not found. Please get the response from the model first.")
    
    try:
        with open(save_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Dataset {dataset_name} with response model {response_model_name} not found")
    
    return data

def load_dataset(save_dir, dataset_name, response_model_name):
    '''
    Load the dataset from the local directory.
    Args:
        save_dir: str, the directory to save the dataset
        dataset_name: str, the name of the dataset
        response_model_name: str, the name of the model to get the response
    Returns: 
        new_dataset: list [
            {
                "instruction": str,
                "output": str,
                "category": str,
            },
        ]
    '''
    metadata = load_metadata(save_dir, dataset_name)
    data = load_response(save_dir, dataset_name, response_model_name)
    
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