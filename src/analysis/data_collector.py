'''
Perturbe the data to collect the data for the analysis.

1. load the dataset
2. load the bias
3. perturb the data
4. get evaluation results
5. save the data
'''
import logging
import os
import json
from datetime import datetime
import asyncio

from abc import ABC, abstractmethod
from src.llm_evaluator import JudgeModelABC, JudgeType, load_judge_model
from src.data.data_utils import load_dataset, load_dataset_for_exploration
from src.evolve_agent.bias_strategies import BiasModification, Bias_types
from src.llm_zoo import load_model
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)

class DataCollector(ABC):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def collect_data(self):
        '''
        Collect the data for the analysis.
        Return:
            - data pairs: list of data dict 
            [{question, init_response, modified_response, bias_strategy_list, init_score, modified_score}, ...]
        '''
        pass

    def save_data(self):
        pass


class PointwiseDataCollector(DataCollector):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
    
    def load_data(self, dataset_name: str, response_model_name: str, judge_backbone: str):
        try:
            dataset = load_dataset_for_exploration(self.data_dir, dataset_name, response_model_name, judge_backbone)
        except Exception as e:
            raise ValueError(f"Dataset {dataset_name} with response model {response_model_name} and judge model {judge_backbone} not found. Evaluate the dataset for exploration first.")
        return dataset

    async def perturb_data(self, dataset: list[dict], judge_backbone: str, helper_model_name: str):
        # learn from UCB
        # 1. modify the response with each of the bias strategies
        # 2. get the scores
        # 3. save the data

        # 1. init the data pairs
        data_pairs = []

        # 2. load the dataset
        question_list = [item['instruction'] for item in dataset]
        init_response_list = [item['output'] for item in dataset]
        original_score_list = [item['original_score'] for item in dataset]
        category_list = [item['category'] for item in dataset]

        #!TODO: whether we need to exclude the data with perfect score?
        
        # 3. init helper functions
        helper_model = load_model(helper_model_name)
        bias_modifier = BiasModification(helper_model)
        llm_evaluator = load_judge_model(JudgeType.POINTWISE, judge_backbone)

        # 4. perturb the data with each of the bias strategies
        for bias in Bias_types:
            modified_response_list = await bias_modifier.batch_principle_guided_mutation(init_response_list, [bias]*len(init_response_list))
            modified_score_list, _ = await llm_evaluator.batch_get_score(question_list, modified_response_list)
            for question, init_response, original_score, modified_response, modified_score, category in zip(question_list, init_response_list, original_score_list, modified_response_list, modified_score_list, category_list):
                data_pairs.append({
                    'instruction': question,
                    'init_response': init_response,
                    'modified_response': modified_response,
                    'bias_strategy': bias,
                    'init_score': original_score,
                    'modified_score': modified_score,
                    'category': category,
                })
        
        return data_pairs
    
    def save_data(self, data_pairs: list[dict], dataset_name: str, response_model_name: str, judge_backbone: str, helper_model_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data2save = {
            "judge_type": "pointwise",
            "dataset_name": dataset_name,
            "response_model_name": response_model_name,
            "judge_backbone": judge_backbone,
            "helper_model_name": helper_model_name,
            "timestamp": timestamp,
            "data_pairs": data_pairs,
        }
        save_path = os.path.join(self.data_dir, dataset_name, f"analysis_{timestamp}.json")
        with open(save_path, "w") as f:
            json.dump(data2save, f)

    async def collect_data(self, dataset_name: str, response_model_name: str, judge_backbone: str, helper_model_name: str):
        logger.info(f"Collecting data for {dataset_name} with response model {response_model_name} and judge model {judge_backbone} and helper model {helper_model_name}")
        dataset = self.load_data(dataset_name, response_model_name, judge_backbone)
        logger.info(f"Loaded {len(dataset)} data points")
        data_pairs = await self.perturb_data(dataset, judge_backbone, helper_model_name)
        logger.info(f"Perturbed {len(data_pairs)} data points")
        self.save_data(data_pairs, dataset_name, response_model_name, judge_backbone, helper_model_name)
        logger.info(f"Saved data")
        return data_pairs
    

class PairwiseDataCollector(DataCollector):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
    
    async def load_data(self, dataset_name: str, response_model_name: str, judge_backbone: str, baseline_model_name: str, answer_position: str = "first"):
        try:
            dataset = load_dataset(self.data_dir, dataset_name, response_model_name)
            baseline_dataset = load_dataset(self.data_dir, dataset_name, baseline_model_name)
            
            # zip two datasets
            inst_mapping = {item['instruction']: item for item in baseline_dataset}
            new_dataset = []
            for item in dataset:
                new_item = item.copy()
                new_item.update({
                    "baseline_response": inst_mapping[item['instruction']]['output'],
                })
                new_dataset.append(new_item.copy())
        except Exception as e:
            raise ValueError(f"Dataset {dataset_name} with response model {response_model_name} and baseline model {baseline_model_name} not found. Please get the response from the model first.")
        
        question_list = [item['instruction'] for item in new_dataset]
        response_list = [item['output'] for item in new_dataset]
        baseline_response_list = [item['baseline_response'] for item in new_dataset]
        
        llm_evaluator = load_judge_model(JudgeType.PAIRWISE, judge_backbone)
        pairwise_scores, _ = await llm_evaluator.batch_get_score(question_list, response_list, baseline_response_list, answer_position=answer_position)
        
        for item, pairwise_score in zip(new_dataset, pairwise_scores):
            item.update({
                "original_score": pairwise_score,
            })
        
        return new_dataset
    
    async def perturb_data(self, dataset: list[dict], judge_backbone: str, helper_model_name: str, answer_position: str = "first"):
        # 0. init the data pairs
        data_pairs = []
        
        # 1. load the dataset
        question_list = [item['instruction'] for item in dataset]
        response_list = [item['output'] for item in dataset]
        baseline_response_list = [item['baseline_response'] for item in dataset]
        original_score_list = [item['original_score'] for item in dataset]
        category_list = [item['category'] for item in dataset]
        
        # 2. init helper functions
        helper_model = load_model(helper_model_name)
        bias_modifier = BiasModification(helper_model)
        llm_evaluator = load_judge_model(JudgeType.PAIRWISE, judge_backbone)
        
        # 3. perturb the data with each of the bias strategies
        for bias in Bias_types:
            modified_response_list = await bias_modifier.batch_principle_guided_mutation(response_list, [bias]*len(response_list))
            modified_score_list, _ = await llm_evaluator.batch_get_score(question_list, modified_response_list, baseline_response_list, answer_position=answer_position)
            for init_response, baseline_response, original_score, modified_response, modified_score, category in zip(response_list, baseline_response_list, original_score_list, modified_response_list, modified_score_list, category_list):
                data_pairs.append({
                    'init_response': init_response,
                    'baseline_response': baseline_response,
                    'modified_response': modified_response,
                    'bias_strategy': [bias],
                    'original_score': original_score,
                    'modified_score': modified_score,
                    'category': category,
                })
        return data_pairs
    
    def save_data(self, data_pairs: list[dict], dataset_name: str, response_model_name: str, judge_backbone: str, baseline_model_name: str, helper_model_name: str, answer_position: str = "first"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data2save = {
            "judge_type": "pairwise",
            "dataset_name": dataset_name,
            "response_model_name": response_model_name,
            "judge_backbone": judge_backbone,
            "baseline_model_name": baseline_model_name,
            "helper_model_name": helper_model_name,
            "answer_position": answer_position,
            "timestamp": timestamp,
            "data_pairs": data_pairs,
        }
        save_path = os.path.join(self.data_dir, dataset_name, f"analysis_{timestamp}.json")
        with open(save_path, "w") as f:
            json.dump(data2save, f)

    async def collect_data(self, dataset_name: str, response_model_name: str, judge_backbone: str, baseline_model_name: str, helper_model_name: str, answer_position: str = "first"):
        dataset = await self.load_data(dataset_name, response_model_name, judge_backbone, baseline_model_name, answer_position)
        data_pairs = await self.perturb_data(dataset, judge_backbone, helper_model_name, answer_position)
        self.save_data(data_pairs, dataset_name, response_model_name, judge_backbone, baseline_model_name, helper_model_name, answer_position)
        return data_pairs


def load_analysis_data_from_trajectories(trajectory_dir: str, dataset_name: str, judge_type: str, judge_backbone: str, response_model_name: str, helper_model_name: str, reward_type:str, baseline_model_name=None, answer_position=None):
    files = os.listdir(trajectory_dir)

    match_files = []
    for file in files:
        path = os.path.join(trajectory_dir, file)
        with open(path, "r") as f:
            data = json.load(f)
        # check match
        if 'reward_type' not in data:
            continue
        if dataset_name == data['dataset_name'] and reward_type == data['reward_type'] and judge_type == data['judge_type'] and judge_backbone == data['judge_backbone'] and response_model_name == data['response_model_name'] and helper_model_name == data['llm_agent_name'] and baseline_model_name == data['baseline_response_model_name'] and answer_position == data['answer_position']:
            logger.info(f"Find {file} for {dataset_name} with judge type {judge_type}, judge backbone {judge_backbone}, response model {response_model_name}, helper model {helper_model_name}, baseline model {baseline_model_name}, and answer position {answer_position}")
            match_files.append(path)
    if len(match_files) == 0:
        raise ValueError(f"No data found for {dataset_name} with judge type {judge_type}, judge backbone {judge_backbone}, response model {response_model_name}, helper model {helper_model_name}, baseline model {baseline_model_name}, and answer position {answer_position}")
    

    init_answer_list = []
    init_score_list = []
    strategy_list = []
    final_answer_list = []
    final_score_list = []
    # use the lastest file
    for path in match_files:
        with open(path, "r") as f:
            data = json.load(f)
        trajectories = data['trajectories']
        for full_trajectory in trajectories:
            trajectory = full_trajectory['best_path']
            len_trajectory = len(trajectory)
            if len_trajectory <= 2:
                continue
            init_step = trajectory[1]
            init_score, _, init_answer, _ = init_step
            current_strategy_list = []
            for step in trajectory[2:]:
                score, _, answer, _ = step
                current_strategy_list.append(step[3])
                strategy_list.append(current_strategy_list.copy())
                init_answer_list.append(init_answer)
                init_score_list.append(init_score)
                final_answer_list.append(answer)
                final_score_list.append(score)
        
    new_data_pairs = []
    for init_answer, init_score, strategy, final_answer, final_score in zip(init_answer_list, init_score_list, strategy_list, final_answer_list, final_score_list):
        new_data_pairs.append({
            'init_response': init_answer,
            'modified_response': final_answer,
            'bias_strategy': strategy,
            'init_score': init_score,
            'modified_score': final_score,
        })
    return new_data_pairs
        
# TODO: support more judges
def load_analysis_data_from_perturbation(data_dir: str, dataset_name: str, judge_type: str, judge_backbone: str, response_model_name: str, helper_model_name: str, baseline_model_name=None, answer_position=None):
    save_path = os.path.join(data_dir, dataset_name)
    files = os.listdir(save_path)

    match_files = []
    for file in files:
        if file[:len("analysis_")] == "analysis_":
            with open(os.path.join(save_path, file), "r") as f:
                data = json.load(f)
                if data["judge_type"] == judge_type and data["judge_backbone"] == judge_backbone and data["response_model_name"] == response_model_name and data["helper_model_name"] == helper_model_name and (baseline_model_name is None or data["baseline_model_name"] == baseline_model_name) and (answer_position is None or data["answer_position"] == answer_position):
                    match_files.append(file)
    
    if len(match_files) == 0:
        raise ValueError(f"No data found for {dataset_name} with judge type {judge_type}, judge backbone {judge_backbone}, response model {response_model_name}, helper model {helper_model_name}, baseline model {baseline_model_name}, and answer position {answer_position}")
    
    # use the lastest file
    logger.info(f"Find {len(match_files)} files for {dataset_name} with judge type {judge_type}, judge backbone {judge_backbone}, response model {response_model_name}, helper model {helper_model_name}, baseline model {baseline_model_name}, and answer position {answer_position}")
    logger.info(f"Load the latest file {match_files[-1]}")
    with open(os.path.join(save_path, match_files[-1]), "r") as f:
        data = json.load(f)
    return data["data_pairs"]

def load_analysis_data(data_dir: str, data_type: str, dataset_name: str, judge_type: str, judge_backbone: str, response_model_name: str, helper_model_name: str, reward_type=None, baseline_model_name=None, answer_position=None):
    if data_type == "perturbation":
        return load_analysis_data_from_perturbation(data_dir, dataset_name, judge_type, judge_backbone, response_model_name, helper_model_name, baseline_model_name=baseline_model_name, answer_position=answer_position)
    elif data_type == "trajectory":
        return load_analysis_data_from_trajectories(data_dir, dataset_name, judge_type, judge_backbone, response_model_name, helper_model_name, reward_type=reward_type, baseline_model_name=baseline_model_name, answer_position=answer_position)
    else:
        raise ValueError(f"Invalid data type: {data_type}")


async def main():
    setup_logging(task_name="collect_data4analysis")
    prefix = "/data2/xianglin"
    
    data_dir = prefix + "/llm-as-a-judge-attack/data"
    dataset_name = "AlpacaEval"
    response_model_name = "gpt-4o-mini"
    judge_backbone = "gemini-2.0-flash"
    helper_model_name = "gpt-4.1-nano"

    data_collector = PointwiseDataCollector(data_dir)
    logger.info(f"Collecting data for {dataset_name} with response model {response_model_name} and judge model {judge_backbone} and helper model {helper_model_name}")
    data_pairs = await data_collector.collect_data(dataset_name, response_model_name, judge_backbone, helper_model_name)
    logger.info(f"Collected {len(data_pairs)} data points")
    print(data_pairs[0])
    

if __name__ == "__main__":
    asyncio.run(main())