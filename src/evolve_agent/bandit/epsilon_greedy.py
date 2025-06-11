import numpy as np
import random
import logging
import argparse
from tqdm import tqdm
import os
import json

from src.evolve_agent.bandit.base import ContextualLinBanditAgent
from src.llm_evaluator import JudgeModel
from src.text_encoder import TextEncoder, MiniLMTextEncoder
from src.llm_zoo import ModelWrapper, load_model
from src.data import load_dataset
from src.logging_utils import setup_logging
from src.data import CATEGORIES

logger = logging.getLogger(__name__)

class ContextualLinEpsilonGreedyAgent(ContextualLinBanditAgent):
    def __init__(self, epsilon: float, n_features: int, llm_agent: ModelWrapper, embedding_model: TextEncoder, llm_evaluator: JudgeModel, reward_type: str = "relative", lambda_reg: float = 1.0):
        """
        Initializes the LinUCB agent.

        Args:
            n_features (int): Dimension of context features (d).
            llm_evaluator (JudgeModel): LLM evaluator to evaluate the quality of the response.
            alpha (float): Exploration parameter. Controls the width of the confidence interval.
                           Higher alpha means more exploration.
            lambda_reg (float): Regularization parameter for Ridge Regression.
                                This is the 'lambda' in (X^T X + lambda*I)^-1 X^T y.
                                Corresponds to initializing A_a with lambda_reg * I.
        """
        super().__init__(n_features, llm_agent, embedding_model, llm_evaluator, reward_type, lambda_reg)
        self.epsilon = epsilon
        self.init_policy_model()

    def predict(self, context_x):
        """
        Predicts the epsilon-greedy score for each arm given the context.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            np.array: A (n_arms x 1) vector of UCB scores for each arm.
        """
        if context_x.shape != (self.n_features, 1):
            raise ValueError(f"Context_x must be a column vector of shape ({self.n_features}, 1)")

        p_ta_values = np.zeros((self.n_arms, 1))

        for arm_idx in range(self.n_arms):
            A_a_inv = np.linalg.inv(self.A[arm_idx])
            theta_hat_a = A_a_inv @ self.b[arm_idx] # (d x d) @ (d x 1) = (d x 1)

            # Expected reward part: x^T * theta_hat_a
            expected_reward = context_x.T @ theta_hat_a # (1 x d) @ (d x 1) = (1 x 1)

            p_ta_values[arm_idx] = expected_reward

        return p_ta_values
     
    def choose_arm(self, context_x):
        """
        Chooses an arm based on the highest UCB score.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            int: The index of the chosen arm.
        """
        prob = np.random.rand()
        if prob<self.epsilon:
            chosen_arm_idx = random.choice(range(self.n_arms))
        else:
            scores = self.predict(context_x)
            chosen_arm_idx = np.argmax(scores)
        return chosen_arm_idx



if __name__ == "__main__":
    setup_logging(task_name="Epsilon-Greedy")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--Budget", type=int, default=20)
    parser.add_argument("--pool_size", type=int, default=3)
    parser.add_argument("--judge_model_name", type=str, default="gemini-1.5-flash")
    parser.add_argument("--llm_agent_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--n_features", type=int, default=384)
    parser.add_argument("--save_path", type=str, default="results/")
    parser.add_argument("--test_mode", type=str, default="policy", choices=["random", "policy"])
    parser.add_argument("--dataset_name", type=str, default="AlpacaEval")
    parser.add_argument("--response_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--data_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data")
    parser.add_argument("--eval_num", type=int, default=5)
    parser.add_argument("--reward_type", type=str, default="relative", choices=["relative", "absolute"])

    args = parser.parse_args()

    dataset_name = args.dataset_name
    epsilon = args.epsilon
    response_model_name = args.response_model_name

    n_features = args.n_features
    llm_agent = load_model(args.llm_agent_name)
    embedding_model = MiniLMTextEncoder()
    judge_model_name = args.judge_model_name
    llm_evaluator = JudgeModel(judge_model_name)
    lambda_reg = args.lambda_reg
    eval_num = args.eval_num

    budget = args.Budget
    pool_size = args.pool_size
    data_dir = args.data_dir
    reward_type = args.reward_type

    logger.info(f"Dataset: {dataset_name}, Response model: {response_model_name}")
    logger.info(f"Budget: {budget}, Pool size: {pool_size}")
    logger.info(f"Epsilon: {epsilon}, Lambda reg: {lambda_reg}")
    logger.info(f"N features: {n_features}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Test mode: {args.test_mode}")
    logger.info(f"Reward type: {reward_type}")

    dataset = load_dataset(data_dir, dataset_name, response_model_name)
    dataset_len = len(dataset)
    logger.info(f"Loaded {dataset_len} questions from {dataset_name}")
    if eval_num > dataset_len:
        eval_num = dataset_len
    else:
        dataset = random.sample(dataset, eval_num)
        logger.info(f"Randomly sample {eval_num} questions from {dataset_len} questions")
    logger.info("-"*100)


    test_results = []
    for idx in tqdm(range(len(dataset))):
        question, response, category = dataset[idx]['instruction'], dataset[idx]['output'], dataset[idx]['category']
        logger.info(f"Question {idx}: {question}")
        logger.info(f"Response {idx}: {response}")

        agent = ContextualLinEpsilonGreedyAgent(epsilon, n_features, llm_agent, embedding_model, llm_evaluator, reward_type, lambda_reg)

        original_score, original_explanation = llm_evaluator.pointwise_score(question, response)
        logger.info(f"Original score: {original_score}, explanation: {original_explanation}")
        if original_score == 9:
            logger.info(f"Perfect score 9, skip")
            # record the results
            test_results.append({
                "category": category,
                "instruction": question,
                "output": response,
                "original_score": original_score,
                "original_explanation": original_explanation,
                "final_score": original_score,
                "final_explanation": original_explanation,
                "final_response": response,
                "skip": True,
            })
            continue

        if args.test_mode == "random":
            final_trajectory = agent.explore_with_random_arm(question, response, pool_size, budget, cold_start=True)
        elif args.test_mode == "policy":
            final_trajectory = agent.explore(question, response, pool_size, budget, cold_start=True)
        else:
            raise ValueError(f"Invalid test mode: {args.test_mode}")
        
        final_score, final_explanation, final_response = final_trajectory[-1]
        exploration_length = len(final_trajectory)-1

        logger.info(f"Final score: {final_score}, explanation: {final_explanation}")
        logger.info(f"Final response: {final_response}")
        logger.info("-"*100)

        # record the results
        test_results.append({
            "category": category,
            "instruction": question,
            "output": response,
            "original_score": original_score,
            "original_explanation": original_explanation,
            "final_score": final_score,
            "final_explanation": final_explanation,
            "final_response": final_response,
            "skip": False,
        })

    # record the evaluation results
    analysis = {
        "strategy": "Epsilon-Greedy",
        "dataset_name": dataset_name,
        "response_model_name": response_model_name,
        "test_mode": args.test_mode,
        "judge_model_name": judge_model_name,
        "llm_agent_name": llm_agent.model_name,
        "lambda_reg": lambda_reg,
        "n_features": n_features,
        "budget": budget,
        "pool_size": pool_size,
        "eval_num": eval_num,
        "reward_type": reward_type,
    }
    # Analyze the test result for each category
    for category in CATEGORIES:
        category_results = [result for result in test_results if result["category"] == category]
        logger.info(f"Category: {category}")
        logger.info(f"Number of results: {len(category_results)}")

        up_num = len([result for result in category_results if result["original_score"] < result["final_score"] and not result["skip"]])
        down_num = len([result for result in category_results if result["original_score"] > result["final_score"] and not result["skip"]])
        tie_num = len([result for result in category_results if result["original_score"] == result["final_score"] and not result["skip"]])
        avg_score_before = np.mean([result["original_score"] for result in category_results])
        avg_score_after = np.mean([result["final_score"] for result in category_results])
        logger.info(f"Number of up results: {up_num}")
        logger.info(f"Number of down results: {down_num}")
        logger.info(f"Number of tie results: {tie_num}")
        logger.info(f"Number of skip results: {len([result for result in category_results if result['skip']])}")
        logger.info(f"Average improvement: {np.mean([result['final_score'] - result['original_score'] for result in category_results if not result['skip']])}")
        logger.info(f"Average score before: {avg_score_before}, average score after: {avg_score_after}")
        logger.info("--------------------------------")

        analysis[category] = {
            "up_num": up_num,
            "down_num": down_num,
            "tie_num": tie_num,
            "skip_num": len([result for result in category_results if result["skip"]]),
            "average_improvement": np.mean([result["final_score"] - result["original_score"] for result in category_results]),
            "avg_score_before": avg_score_before,
            "avg_score_after": avg_score_after,
        }
    
    # save the analysis in the output
    save_path = os.path.join(args.save_path, f"evaluation_results.json")
    if os.path.exists(save_path):
        history_analysis = json.load(open(save_path, "r"))
        history_analysis.append(analysis)
    else:
        history_analysis = [analysis]
        
    with open(save_path, "w") as f:
        json.dump(history_analysis, f)
    