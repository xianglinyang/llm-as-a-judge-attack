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
import logging
import asyncio
import argparse

from src.logging_utils import setup_logging
from src.llm_evaluator import load_judge_model, get_judge_type
from src.data.data_utils import load_dataset
from src.llm_zoo.imp2name import get_model_name, is_valid_model


logger = logging.getLogger(__name__)

async def get_evaluation_score(data_dir, dataset_name, response_model_implementation_name, judge_model_implementation_name, judge_type):
    logger.info(f"Processing {dataset_name} with response model {response_model_implementation_name} and judge model {judge_model_implementation_name}")

    if not is_valid_model(response_model_implementation_name):
        raise ValueError(f"Model {response_model_implementation_name} is not valid!")
    
    response_model_name = get_model_name(response_model_implementation_name)
    judge_model_name = get_model_name(judge_model_implementation_name)

    # save the dataset
    save_path = os.path.join(data_dir, dataset_name, f"dataset_for_exploration_{response_model_name}_{judge_model_name}_{judge_type}.json")
    if os.path.exists(save_path):
        logger.info(f"Dataset {save_path} already exists. Skipping evaluation.")
        return

    judge_type_enum = get_judge_type(judge_type)

    llm_evaluator = load_judge_model(judge_type_enum, judge_model_implementation_name)
    dataset = load_dataset(data_dir, dataset_name, response_model_name)
    dataset_len = len(dataset)
    logger.info(f"Loaded {dataset_len} questions from {dataset_name} with response model {response_model_name} and judge model {judge_model_name}")

    # preprocess the dataset
    question_list = [item['instruction'] for item in dataset]
    response_list = [item['output'] for item in dataset]
    original_score_list, original_explanation_list = await llm_evaluator.batch_get_score(question_list, response_list)

    # construct the dataset for exploration
    dataset_for_exploration = []
    for item, original_score, original_explanation in zip(dataset, original_score_list, original_explanation_list):
        item.update({
            'original_score': original_score,
            'original_explanation': original_explanation,
        })
        dataset_for_exploration.append(item.copy())
    
    # save the dataset
    with open(save_path, "w") as f:
        json.dump(dataset_for_exploration, f, indent=4)
    logger.info(f"Saved dataset to {save_path}")

    # logging statistics: How many 9 scores
    logger.info(f"How many 10 scores: {sum(1 for score in original_score_list if score >= 10)}")
    logger.info(f"How many 9 scores: {sum(1 for score in original_score_list if score == 9)}")
    logger.info(f"How many 8 scores: {sum(1 for score in original_score_list if score == 8)}")
    logger.info(f"How many 7 scores: {sum(1 for score in original_score_list if score == 7)}")
    logger.info(f"How many 6 scores: {sum(1 for score in original_score_list if score == 6)}")
    logger.info(f"How many 5 scores: {sum(1 for score in original_score_list if score == 5)}")
    logger.info(f"How many 4 scores: {sum(1 for score in original_score_list if score == 4)}")
    logger.info(f"How many 3 scores: {sum(1 for score in original_score_list if score == 3)}")
    logger.info(f"How many 2 scores: {sum(1 for score in original_score_list if score == 2)}")
    logger.info(f"How many 1 scores: {sum(1 for score in original_score_list if score == 1)}")
    logger.info(f"How many -1 scores: {sum(1 for score in original_score_list if score == -1)}")
    
    # save the dataset
    save_path = os.path.join(data_dir, dataset_name, f"dataset_for_exploration_{response_model_name}_{judge_model_name}.json")
    with open(save_path, "w") as f:
        json.dump(dataset_for_exploration, f, indent=4)
    logger.info(f"Saved dataset to {save_path}")

    return dataset_for_exploration


async def main(args):
    setup_logging(task_name="get_evaluation_score")

    data_dir = args.data_dir
    dataset_name = args.dataset_name
    response_model_name = args.response_model_name
    judge_model_name = args.judge_model_name
    judge_type = args.judge_type

    dataset_for_exploration = await get_evaluation_score(data_dir, dataset_name, response_model_name, judge_model_name, judge_type)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process")
    parser.add_argument("--response_model_name", type=str, required=True, help="Implementation name of the model to generate responses")
    parser.add_argument("--judge_model_name", type=str, required=True, help="Implementation name of the model to judge")
    parser.add_argument("--judge_type", type=str, required=True, help="Type of the judge model", choices=["pointwise", "pairwise", "pairwise_fine_grained", "alpaca_eval", "arena_hard_auto", "mt_bench", "mlr_bench"])
    args = parser.parse_args()
    asyncio.run(main(args))

    # name = "openai/o3"
    # c = get_model_name(name)
    # print(c)