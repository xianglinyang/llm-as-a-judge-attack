# parse the data_path.yaml
import yaml
import json

def load_data_path():
    with open("/home/ljiahao/xianglin/git_space/llm-as-a-judge-attack/src/data_path.yaml", "r") as f:
        data_path = yaml.load(f, Loader=yaml.FullLoader)
    return data_path

def load_dataset(dataset_name, response_model_name):
    data_path = load_data_path()
    try:    
        path = data_path[dataset_name][response_model_name]['path']
        with open(path, 'r') as f:
            data = json.load(f)
    except:
        raise ValueError(f"Dataset {dataset_name} with response model {response_model_name} not found")
    return data

def assign_category(dataset):
    pass

def get_response(dataset, response_model_name):
    pass

if __name__ == "__main__":
    dataset = load_dataset("lima", "human_written")
    print(dataset[0])
    print(len(dataset))
    
