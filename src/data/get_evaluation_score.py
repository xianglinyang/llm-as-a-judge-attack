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
import logging


from src.logging_utils import setup_logging
from src.llm_evaluator import load_judge_model, JudgeType
from src.data.data_utils import load_dataset
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    data_dir = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"
    
    # ------------------------------------------------------------
    # get score from judge model
    # ------------------------------------------------------------
    setup_logging(task_name="preprocess_dataset_for_exploration")
    
    judge_model_list = [
        "gemini-2.0-flash",
        # "gpt-4o",
        # "gpt-4.1",
    ]
    dataset_list = [
        "AlpacaEval",
        # "ArenaHard",
        # "MTBench",
    ]
    response_model_list = [
        # "gpt-4o-mini"
        # "gpt-4.1-mini",
        # "gpt-4.1-nano",
        # "gemini-1.5-flash-8b",
        "gpt-4o-2024-05-13"
    ]

    judge_type = JudgeType.POINTWISE

    for dataset_name in dataset_list:
        for judge_model_name in judge_model_list:
            for response_model_name in response_model_list:
                logger.info(f"Processing {dataset_name} with response model {response_model_name} and judge model {judge_model_name}")

                llm_evaluator = load_judge_model(judge_type, judge_model_name)
                dataset = load_dataset(data_dir, dataset_name, response_model_name)
                dataset_len = len(dataset)
                logger.info(f"Loaded {dataset_len} questions from {dataset_name} with response model {response_model_name} and judge model {judge_model_name}")

                # preprocess the dataset
                question_list = [item['instruction'] for item in dataset]
                response_list = [item['output'] for item in dataset]
                original_score_list, original_explanation_list = llm_evaluator.batch_get_score(question_list, response_list)

                # construct the dataset for exploration
                dataset_for_exploration = []
                for item, original_score, original_explanation in zip(dataset, original_score_list, original_explanation_list):
                    item.update({
                        'original_score': original_score,
                        'original_explanation': original_explanation,
                    })
                    dataset_for_exploration.append(item.copy())
                
                # save the dataset
                save_path = os.path.join(data_dir, dataset_name, f"dataset_for_exploration_{response_model_name}_{judge_model_name}.json")
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
                