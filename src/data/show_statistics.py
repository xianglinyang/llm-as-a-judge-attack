import logging

from src.data.data_utils import load_dataset_for_exploration
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    data_dir = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data"
    setup_logging(task_name="show_statistics")
    
    dataset_name = "MTBench"
    response_model_name = "gpt-4o-mini"
    judge_model_name = "gemini-2.0-flash"

    dataset = load_dataset_for_exploration(data_dir, dataset_name, response_model_name, judge_model_name)
    
    # logging statistics: How many 9 scores
    logger.info(f"Length of the dataset: {len(dataset)}")
    logger.info(f"keys of the dataset: {dataset[0].keys()}")
    logger.info(f"How many 9 scores: {sum(1 for item in dataset if item['original_score'] == 9)}")
    logger.info(f"How many 8 scores: {sum(1 for item in dataset if item['original_score'] == 8)}")
    logger.info(f"How many 7 scores: {sum(1 for item in dataset if item['original_score'] == 7)}")
    logger.info(f"How many 6 scores: {sum(1 for item in dataset if item['original_score'] == 6)}")
    logger.info(f"How many 5 scores: {sum(1 for item in dataset if item['original_score'] == 5)}")
    logger.info(f"How many 4 scores: {sum(1 for item in dataset if item['original_score'] == 4)}")
    logger.info(f"How many 3 scores: {sum(1 for item in dataset if item['original_score'] == 3)}")
    logger.info(f"How many 2 scores: {sum(1 for item in dataset if item['original_score'] == 2)}")
    logger.info(f"How many 1 scores: {sum(1 for item in dataset if item['original_score'] == 1)}")
    logger.info(f"How many -1 scores: {sum(1 for item in dataset if item['original_score'] == -1)}")

    # Example element
    logger.info(f"Example element:")
    logger.info(f"Question: {dataset[0]['instruction']}")
    logger.info(f"Original Response: {dataset[0]['output']}")
    logger.info(f"Original Score: {dataset[0]['original_score']}")
    logger.info(f"Original Explanation: {dataset[0]['original_explanation']}")
    logger.info(f"Category: {dataset[0]['category']}")