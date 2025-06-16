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
from src.llm_evaluator import JudgeModel
from src.data.data_utils import load_dataset
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    data_dir = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"
    
    # ------------------------------------------------------------
    # get score from judge model and filter out the perfect score 9 samples
    # ------------------------------------------------------------
    setup_logging(task_name="preprocess_dataset_for_exploration")
    
    judge_model_list = [
        "gemini-1.5-flash",
        "gpt-4o",
        "gpt-4.1",
    ]
    dataset_list = [
        "AlpacaEval",
        "ArenaHard",
        "MTBench",
    ]
    response_model_list = [
        "gpt-4o-mini"
    ]

    for dataset_name in dataset_list:
        for judge_model_name in judge_model_list:
            for response_model_name in response_model_list:
                logger.info(f"Processing {dataset_name} with response model {response_model_name} and judge model {judge_model_name}")

                llm_evaluator = JudgeModel(model_name=judge_model_name)
                dataset = load_dataset(data_dir, dataset_name, response_model_name)
                dataset_len = len(dataset)
                logger.info(f"Loaded {dataset_len} questions from {dataset_name} with response model {response_model_name} and judge model {judge_model_name}")

                # preprocess the dataset
                question_list = [item['instruction'] for item in dataset]
                response_list = [item['output'] for item in dataset]
                category_list = [item['category'] for item in dataset]
                original_score_list, original_explanation_list = asyncio.run(llm_evaluator.batch_pointwise_score(question_list, response_list))

                # construct the dataset for exploration
                dataset_for_exploration = [
                    {
                        'instruction': question,
                        'output': response,
                        'category': category,
                        'original_score': original_score,
                        'original_explanation': original_explanation,
                    }
                    for question, response, category, original_score, original_explanation in zip(question_list, response_list, category_list, original_score_list, original_explanation_list)
                ]
                
                # save the dataset
                save_path = os.path.join(data_dir, dataset_name, f"dataset_for_exploration_{response_model_name}_{judge_model_name}.json")
                with open(save_path, "w") as f:
                    json.dump(dataset_for_exploration, f, indent=4)
                logger.info(f"Saved dataset to {save_path}")
                