import logging
import numpy as np
import os
import json
import time
import random
from typing import Dict, List

from src.data.data_utils import load_dataset_for_exploration
from src.llm_evaluator import JudgeType, load_judge_model
from src.llm_zoo.api_zoo import get_model_name

logger = logging.getLogger(__name__)

def find_shortest_of_max_simple(data):
    if not data:
        return None

    # 1. First pass: Find the maximum key value.
    max_key = max(item[0] for item in data)
    
    # 2. Second pass: Create a new list of all items that have that max key.
    all_max_items = [item for item in data if item[0] == max_key]
    
    # 3. On this smaller list, find the item with the minimum length.
    return min(all_max_items, key=len)

def find_longest_of_min_simple(data):
    if not data:
        return None
    
    # 1. First pass: Find the minimum key value.
    min_key = min(item[0] for item in data)

    # 2. Second pass: Create a new list of all items that have that min key.
    all_min_items = [item for item in data if item[0] == min_key]
    
    # 3. On this smaller list, find the item with the maximum length.
    return max(all_min_items, key=len)

def find_longest_of_min_simple_index(data):
    if not data:
        return None
    
    # 1. First pass: Find the minimum key value.
    min_key = min(item[0] for item in data)
    
    # 2. Second pass: Create a new list of all items that have that min key.
    all_min_items = [item for item in data if item[0] == min_key]
    
    # 3. On this smaller list, find the item with the maximum length.
    return data.index(max(all_min_items, key=len))

def _estimate_tokens(txt: str) -> int:
    # lightweight token proxy; swap with your tokenizer if available
    # e.g., return self.tokenizer.count_tokens(txt)
    return max(1, len(txt.split()))

def _batch_estimate_tokens(texts: List[str]) -> List[int]:
    # Batch token estimation for better performance
    return [max(1, len(txt.split())) for txt in texts]

def _get_pool_metrics(pool: List[Dict]) -> tuple[float, float]:
    # Get best score and pool mean in single pass
    if not pool:
        return 0.0, 0.0
    
    scores = [item["score"] for item in pool]
    return max(scores), sum(scores) / len(scores)

def _find_pool_extremes(pool: List[Dict]) -> tuple[int, int]:
    # Find both best and worst indices in single pass
    if not pool:
        raise ValueError("Empty pool")
    
    if len(pool) == 1:
        return 0, 0
    
    best_score = worst_score = pool[0]["score"]
    best_i = worst_i = 0
    
    for i, item in enumerate(pool):
        score = item["score"]
        
        # Check for best
        if score > best_score or (score == best_score and len(item['history']) < len(pool[best_i]['history'])):
            best_score = score
            best_i = i
        
        # Check for worst
        if score < worst_score or (score == worst_score and len(item['history']) > len(pool[worst_i]['history'])):
            worst_score = score
            worst_i = i
    
    return best_i, worst_i

def _select_from_pool_uniform(pool: List[Dict]) -> int:
    return random.randrange(len(pool))

def _best_item(pool: List[Dict]) -> Dict:
    # highest score with the shortest history - optimized
    if not pool:
        raise ValueError("Empty pool")
    
    if len(pool) == 1:
        return pool[0]
    
    best_score = float("-inf")
    best_i = 0
    
    for i, item in enumerate(pool):
        score = item["score"]
        if score > best_score or (score == best_score and len(item['history']) < len(pool[best_i]['history'])):
            best_score = score
            best_i = i
    
    return pool[best_i]

def _worst_index(pool: List[Dict]) -> int:
    # index of lowest score - optimized
    if not pool:
        raise ValueError("Empty pool")
    
    if len(pool) == 1:
        return 0
    
    worst_score = float("inf")
    worst_i = 0
    
    for i, item in enumerate(pool):
        score = item["score"]
        if score < worst_score or (score == worst_score and len(item['history']) > len(pool[worst_i]['history'])):
            worst_score = score
            worst_i = i
    
    return worst_i


# TODO: consider the pairwise ordering
async def prepare_dataset_for_exploration(data_dir, dataset_name, response_model_implementation_name, judge_type, judge_implementation_name, baseline_response_model_implementation_name=None, answer_position: str = None):
    '''
    Prepare the dataset for exploration. Consider the judge type and judge backbone.
    Args:
        data_dir: the directory of the dataset
        dataset_name: the name of the dataset
        response_model_name: the name of the response model
        judge_type: the type of the judge
        judge_backbone: the backbone of the judge
    Returns:
        question_list: the list of questions
        init_response_list: the list of initial responses
        original_score_list: the list of original scores
        original_explanation_list: the list of original explanations
        category_list: the list of categories
        baseline_response_list: the list of baseline responses
    '''
    response_model_name = get_model_name(response_model_implementation_name)
    judge_backbone = get_model_name(judge_implementation_name)
    baseline_response_model_name = get_model_name(baseline_response_model_implementation_name) if baseline_response_model_implementation_name else None

    dataset = load_dataset_for_exploration(data_dir, dataset_name, response_model_name, judge_backbone)

    question_list = [item['instruction'] for item in dataset]
    init_response_list = [item['output'] for item in dataset]
    category_list = [item['category'] for item in dataset]

    if judge_type in [JudgeType.POINTWISE, JudgeType.MT_BENCH, JudgeType.MLR_BENCH, JudgeType.POINTWISE_RANDOMIZED, JudgeType.POINTWISE_IGNORE_BIAS]:
        original_score_list = [item['original_score'] for item in dataset]
        original_explanation_list = [item['original_explanation'] for item in dataset]
        baseline_response_list = init_response_list.copy()
    elif judge_type in [JudgeType.PAIRWISE, JudgeType.PAIRWISE_FINE_GRAINED, JudgeType.ALPACA_EVAL, JudgeType.ARENA_HARD_AUTO]:
        assert baseline_response_model_name is not None, "Baseline response model name is required for pairwise evaluation"
        baseline_dataset = load_dataset_for_exploration(data_dir, dataset_name, baseline_response_model_name, judge_backbone)
        baseline_response_list = [item['output'] for item in baseline_dataset]
        assert len(question_list) == len(baseline_response_list), "Question list and baseline response list must have the same length"

        # get pairwise score
        judge_model = load_judge_model(judge_type, judge_implementation_name)
        if answer_position == "first":  
            original_score_list, original_explanation_list = await judge_model.batch_get_score(question_list, init_response_list, baseline_response_list)
        elif answer_position == "second":
            original_score_list, original_explanation_list = await judge_model.batch_get_score(question_list, baseline_response_list, init_response_list)
            original_score_list = [- score for score in original_score_list]
        else:
            raise ValueError(f"Invalid answer position: {answer_position}")
    else:
        raise ValueError(f"Unsupported judge type: {judge_type}")

    return question_list, init_response_list, category_list, original_score_list, original_explanation_list, baseline_response_list


def exclude_perfect_response(judge_type, question_list, init_response_list, category_list, original_score_list, original_explanation_list, baseline_response_list):
    # preprocess the dataset, exclude the perfect score
    test_results = []
    selected_idxs = []

    for idx, (question, response, category, original_score, original_explanation) in enumerate(zip(question_list, init_response_list, category_list, original_score_list, original_explanation_list)):
        if judge_type in [JudgeType.POINTWISE, JudgeType.MLR_BENCH, JudgeType.POINTWISE_RANDOMIZED, JudgeType.POINTWISE_IGNORE_BIAS]:
            if original_score >= 9:
                test_results.append({
                    "category": category,
                    "instruction": question,
                    "output": response,
                    "original_score": original_score,
                    "original_explanation": original_explanation,
                    "final_score": original_score,
                    "final_explanation": original_explanation,
                    "final_response": response,
                    "exploration_length": 1,
                    "skip": 1,
                })
                continue
            elif original_score == -1:
                continue
            else:
                selected_idxs.append(idx)
        elif judge_type in [JudgeType.MT_BENCH]:
            if original_score >= 5:
                test_results.append({
                    "category": category,
                    "instruction": question,
                    "output": response,
                    "original_score": original_score,
                    "original_explanation": original_explanation,
                    "final_score": original_score,
                    "final_explanation": original_explanation,
                    "final_response": response,
                    "exploration_length": 1,
                    "skip": 1,
                })
                continue
            elif original_score == -1:
                continue
            else:
                selected_idxs.append(idx)
        elif judge_type in [JudgeType.PAIRWISE, JudgeType.PAIRWISE_FINE_GRAINED, JudgeType.ALPACA_EVAL, JudgeType.ARENA_HARD_AUTO]:
            # if win, skip
            if original_score > 0:
                test_results.append({
                    "category": category,
                    "instruction": question,
                    "output": response,
                    "baseline_response": baseline_response_list[idx] if baseline_response_list else None,
                    "original_score": original_score,
                    "original_explanation": original_explanation,
                    "final_score": original_score,
                    "final_response": response,
                    "exploration_length": 1,
                    "skip": 1,
                })
            else:
                selected_idxs.append(idx)
        else:
            raise ValueError(f"Unsupported judge type: {judge_type}")
        
    dataset_len = len(question_list)
    selected_idxs_len = len(selected_idxs)
    logger.info(f"Loaded {dataset_len} questions...")
    logger.info(f"Selected {selected_idxs_len} valid questions...")

    return test_results, selected_idxs

def extract_result_from_trajectories(question_list, init_response_list, category_list, original_score_list, original_explanation_list, baseline_response_list, trajectories):
    test_results = []
    for i, (question, response, category, original_score, original_explanation, baseline_response, trajectory_dict) in enumerate(zip(question_list, init_response_list, category_list, original_score_list, original_explanation_list, baseline_response_list, trajectories)):

        trajectory = trajectory_dict["history"]
        final_score, final_explanation, final_response, _ = trajectory[-1]
        exploration_length = len(trajectory) -1
        
        result = {
            "category": category,
            "instruction": question,
            "output": response,
            "original_score": original_score,
            "original_explanation": original_explanation,
            "final_score": final_score,
            "final_explanation": final_explanation,
            "final_response": final_response,
            "baseline_response": baseline_response,
            "exploration_length": exploration_length,
            "skip": 0,
        }
        
        test_results.append(result.copy())
    return test_results

def get_result_analysis(test_results):
    # record the evaluation results
    categories = set([result["category"] for result in test_results])
    analysis = {}
    analysis["categories"] = dict()
    # Analyze the test result for each category
    for category in categories:
        category_results = [result for result in test_results if result["category"] == category]
        logger.info(f"Category: {category}")
        logger.info(f"Number of results: {len(category_results)}")

        # Pointwise evaluation analysis
        up_num = len([1 for result in category_results if (not result["skip"]) and result["original_score"] < result["final_score"]])
        down_num = len([1 for result in category_results if (not result["skip"]) and result["original_score"] > result["final_score"]])
        tie_num = len([1 for result in category_results if (not result["skip"]) and result["original_score"] == result["final_score"]])
        skip_num = len([1 for result in category_results if result["skip"]])
        all_num = len(category_results)
        avg_exploration_length = np.mean([result["exploration_length"] for result in category_results if (not result["skip"])])
        avg_score_before = np.mean([result["original_score"] for result in category_results if (not result["skip"])])
        avg_score_after = np.mean([result["final_score"] for result in category_results if (not result["skip"])])
        avg_improvement = np.mean([result["final_score"] - result["original_score"] for result in category_results if (not result["skip"])])

        logger.info(f"Number of up results: {up_num}")
        logger.info(f"Number of down results: {down_num}")
        logger.info(f"Number of tie results: {tie_num}")
        logger.info(f"Number of skip results: {skip_num}")
        logger.info(f"Number of all results: {all_num}")
        logger.info(f"Average exploration length: {avg_exploration_length}")
        logger.info(f"Average improvement: {avg_improvement}")
        logger.info(f"Average score before: {avg_score_before}, average score after: {avg_score_after}")
        logger.info("--------------------------------")

        analysis["categories"][category] = {
            "up_num": up_num,
            "down_num": down_num,
            "tie_num": tie_num,
            "skip_num": skip_num,
            "all_num": all_num,
            "exploration_length": avg_exploration_length,
            "avg_score_before": avg_score_before,
            "avg_score_after": avg_score_after,
            "average_improvement": avg_improvement,
        }
    return analysis


def save_result_analysis(analysis, save_path):
    # save the analysis in the output
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"evaluation_results.json")
    if os.path.exists(save_path):
        history_analysis = json.load(open(save_path, "r"))
        history_analysis.append(analysis)
    else:
        history_analysis = [analysis]
        
    with open(save_path, "w") as f:
        json.dump(history_analysis, f)
    logger.info(f"Analysis saved to {save_path}")
    logger.info("-"*100)


def save_trajectories(trajectories, save_path, save_name):
    # save the trajectories
    os.makedirs(save_path, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    save_path = os.path.join(save_path, f"{save_name}_{timestamp}.json")
    with open(save_path, "w") as f:
        json.dump(trajectories, f)
    logger.info(f"Trajectories saved to {save_path}")
    logger.info("-"*100)

def save_metrics(metrics, save_path, save_name):
    # save the metrics
    os.makedirs(save_path, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    save_path = os.path.join(save_path, f"{save_name}_{timestamp}.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f)
    logger.info(f"Metrics saved to {save_path}")
    logger.info("-"*100)


def sample_and_filter_data(selected_idxs, eval_num, question_list, init_response_list, 
                              original_score_list, original_explanation_list, category_list, 
                              baseline_response_list):
    available_num = len(selected_idxs)
    if eval_num >= available_num:
        eval_num = available_num
        logger.info(f"Eval num exceed the number of selected idxs, use all selected idxs.")
        logger.info(f"Eval num: {eval_num}")
    else:
        selected_idxs = random.sample(selected_idxs, eval_num)
        logger.info(f"Randomly sample {eval_num} questions from {available_num} questions")

    # selected samples
    question_list = [question_list[idx] for idx in selected_idxs]
    init_response_list = [init_response_list[idx] for idx in selected_idxs]
    original_score_list = [original_score_list[idx] for idx in selected_idxs]
    original_explanation_list = [original_explanation_list[idx] for idx in selected_idxs]
    category_list = [category_list[idx] for idx in selected_idxs]
    baseline_response_list = [baseline_response_list[idx] for idx in selected_idxs]

    return eval_num, selected_idxs, question_list, init_response_list, original_score_list, original_explanation_list, category_list, baseline_response_list


if __name__ == "__main__":
    # test the function
    test_results = [
        {"category": "category1", "original_score": 8, "final_score": 9, "skip": 1, "exploration_length": 1},
        {"category": "category1", "original_score": 8, "final_score": 7, "skip": 1, "exploration_length": 1},
        {"category": "category1", "original_score": 8, "final_score": 8, "skip": 0, "exploration_length": 1},
        {"category": "category1", "original_score": 8, "final_score": 9, "skip": 0, "exploration_length": 1},
        {"category": "category1", "original_score": 8, "final_score": 7, "skip": 0, "exploration_length": 1},
        {"category": "category2", "original_score": 8, "final_score": 9, "skip": 1, "exploration_length": 1},
        {"category": "category2", "original_score": 8, "final_score": 7, "skip": 1, "exploration_length": 1},
        {"category": "category2", "original_score": 8, "final_score": 8, "skip": 0, "exploration_length": 1},
        {"category": "category2", "original_score": 8, "final_score": 9, "skip": 0, "exploration_length": 1},
        {"category": "category2", "original_score": 8, "final_score": 7, "skip": 0, "exploration_length": 1},
    ]
    analysis = get_result_analysis(test_results)
    print(analysis)