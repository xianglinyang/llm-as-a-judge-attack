import numpy as np
import logging
import argparse
import time
import asyncio

from src.evolve_agent.bandit.base import ContextualLinBanditAgent
from src.llm_evaluator import JudgeType
from src.text_encoder import TextEncoder, MiniLMTextEncoder
from src.llm_zoo import ModelWrapper, load_model
from src.logging_utils import setup_logging
from src.evolve_agent.utils import prepare_dataset_for_exploration, exclude_perfect_response, sample_and_filter_data, extract_result_from_trajectories, get_result_analysis, save_result_analysis, save_trajectories
    

logger = logging.getLogger(__name__)

class ContextualLinUCBAgent(ContextualLinBanditAgent):
    def __init__(self, n_features: int, llm_agent: ModelWrapper, embedding_model: TextEncoder, judge_type: JudgeType, judge_model_backbone: str, reward_type: str = "relative", alpha: float = 1.0, lambda_reg: float = 1.0, answer_position: str = "first"):
        """
        Initializes the LinUCB agent.

        Args:
            n_features (int): Dimension of context features (d).
            llm_agent (ModelWrapper): LLM agent to generate the response.
            embedding_model (TextEncoder): Embedding model to encode the context.
            judge_type (JudgeType): Type of judge evaluation (pointwise, pairwise, etc.)
            judge_model_backbone (str): Backbone model for the judge
            reward_type (str): Type of reward to use ("relative" or "absolute").
            alpha (float): Exploration parameter. Controls the width of the confidence interval.
                           Higher alpha means more exploration.
            lambda_reg (float): Regularization parameter for Ridge Regression.
                                This is the 'lambda' in (X^T X + lambda*I)^-1 X^T y.
                                Corresponds to initializing A_a with lambda_reg * I.
            answer_position (str): Position of the answer in pairwise comparison ("first" or "second").
        """
        super().__init__(n_features, llm_agent, embedding_model, judge_type, judge_model_backbone, reward_type, lambda_reg, answer_position)
        self.alpha = alpha
        self.init_policy_model()

    def predict(self, context_x):
        """
        Predicts the UCB score for each arm given the context.

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

            # Exploration bonus part: alpha * sqrt(x^T * A_a_inv * x)
            uncertainty_term = context_x.T @ A_a_inv @ context_x # (1 x d) @ (d x d) @ (d x 1) = (1 x 1)
            exploration_bonus = self.alpha * np.sqrt(uncertainty_term)

            p_ta_values[arm_idx] = expected_reward + exploration_bonus

        return p_ta_values
    
    def batch_predict(self, context_x_list):
        """
        Predicts the UCB score for each arm given the context.
        Args:
            context_x_list (np.array): A (n_samples, n_features, 1) tensor representing the context.
        Returns:
            np.array: A (n_samples, n_arms, 1) tensor of UCB scores for each arm.
        """
        if context_x_list.shape[1:] != (self.n_features, 1):
            raise ValueError(f"Context_x must be a tensor of shape (n_samples, {self.n_features}, 1)")
        
        n_samples = context_x_list.shape[0]
        p_ta_values = np.zeros((n_samples, self.n_arms, 1))
        
        for arm_idx in range(self.n_arms):
            A_a_inv = np.linalg.inv(self.A[arm_idx])
            theta_hat_a = A_a_inv @ self.b[arm_idx]  # (d x d) @ (d x 1) = (d x 1)

            # Reshape context_x_list for batch operations
            context_x_reshaped = context_x_list.reshape(n_samples, self.n_features)  # (n_samples, n_features)
            
            # Expected reward part: x^T * theta_hat_a
            expected_reward = context_x_reshaped @ theta_hat_a  # (n_samples, n_features) @ (n_features, 1) = (n_samples, 1)

            # Exploration bonus part: alpha * sqrt(x^T * A_a_inv * x)
            uncertainty_term = np.sum((context_x_reshaped @ A_a_inv) * context_x_reshaped, axis=1, keepdims=True)  # (n_samples, 1)
            exploration_bonus = self.alpha * np.sqrt(uncertainty_term)

            p_ta_values[:, arm_idx] = expected_reward + exploration_bonus

        return p_ta_values
    
     
    def choose_arm(self, context_x):
        """
        Chooses an arm based on the highest UCB score.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            int: The index of the chosen arm.
        """
        ucb_scores = self.predict(context_x)
        chosen_arm_idx = np.argmax(ucb_scores)
        return chosen_arm_idx
    
    def batch_choose_arm(self, context_x_list):
        """
        Chooses an arm based on the highest UCB score.
        Args:
            context_x_list (np.array): A (n_samples, n_features, 1) tensor representing the context.
        Returns:
            np.array: A (n_samples,) tensor of the chosen arm index for each sample.
        """
        ucb_scores = self.batch_predict(context_x_list)
        chosen_arm_idx = np.argmax(ucb_scores, axis=1)
        return chosen_arm_idx

async def main(args):
    llm_agent = load_model(args.llm_agent_name)
    embedding_model = MiniLMTextEncoder()
    judge_type = JudgeType(args.judge_type)
    judge_model_backbone = args.judge_model_name

    # Use the enhanced reward system instead of manual pairwise scoring
    question_list, init_response_list, category_list, original_score_list, original_explanation_list, baseline_response_list = await prepare_dataset_for_exploration(args.data_dir, args.dataset_name, args.response_model_name, judge_type, judge_model_backbone, args.baseline_response_model_name, args.answer_position)
    test_results, selected_idxs = exclude_perfect_response(judge_type, question_list, init_response_list, category_list, original_score_list, original_explanation_list, baseline_response_list)
    logger.info(f"Skipped {len(test_results)} samples")
    logger.info(f"Dataset for exploration: {len(selected_idxs)} samples...")
    eval_num, selected_idxs, question_list, init_response_list, original_score_list, original_explanation_list, category_list, baseline_response_list = sample_and_filter_data(selected_idxs, args.eval_num, question_list, init_response_list, original_score_list, original_explanation_list, category_list, baseline_response_list)

    logger.info(f"Initializing the agent...")
    agent = ContextualLinUCBAgent(args.n_features, llm_agent, embedding_model, judge_type, judge_model_backbone, args.reward_type, args.alpha, args.lambda_reg, args.answer_position)
    logger.info(f"Agent initialized.")
    logger.info("-"*100)

    start_time = time.time()

    trajectories = []
    if args.test_mode == "single":
        logger.info(f"Running single exploration...")
        trajectories = await agent.batch_explore(question_list, init_response_list, original_score_list, original_explanation_list, args.pool_size, args.Budget, cold_start=True, baseline_response_list=baseline_response_list)
        logger.info(f"Single exploration finished.")
        logger.info("-"*100)
    elif args.test_mode == "random":
        logger.info(f"Running random exploration...")
        trajectories = await agent.batch_explore_with_random_arm(question_list, init_response_list, original_score_list, original_explanation_list, args.pool_size, args.Budget, baseline_response_list=baseline_response_list)
        logger.info(f"Random exploration finished.")
        logger.info("-"*100)
    else:
        raise ValueError(f"Invalid test mode: {args.test_mode}")
    
    new_test_results = extract_result_from_trajectories(question_list, init_response_list, category_list, original_score_list, original_explanation_list, baseline_response_list, trajectories)
    test_results.extend(new_test_results)
    
    end_time = time.time()
        
    analysis = get_result_analysis(test_results)
    meta_info = {
        "strategy": "UCB",
        "judge_type": args.judge_type,
        "answer_position": args.answer_position,
        "dataset_name": args.dataset_name,
        "judge_backbone": judge_model_backbone,
        "baseline_response_model_name": args.baseline_response_model_name,
        "llm_agent_name": args.llm_agent_name,
        "response_model_name": args.response_model_name,
        "test_mode": args.test_mode,
        "lambda_reg": args.lambda_reg,
        "n_features": args.n_features,
        "budget": args.Budget,
        "pool_size": args.pool_size,
        "eval_num": eval_num,
        "reward_type": args.reward_type,
        "alpha": args.alpha,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_taken": end_time - start_time,
    }
    analysis.update(meta_info)
    trajectories_to_save = meta_info.copy()
    trajectories_to_save.update({"trajectories": trajectories})

    save_result_analysis(analysis, args.save_analysis_path)
    save_trajectories(trajectories_to_save, args.save_trajectory_path, f"ucb")

    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds for exploration with {args.Budget} budget and {args.pool_size} pool size and {eval_num} eval num")


if __name__ == "__main__":
    setup_logging(task_name="UCB")

    parser = argparse.ArgumentParser()
    parser.add_argument("--Budget", type=int, default=20)
    parser.add_argument("--pool_size", type=int, default=3)
    parser.add_argument("--judge_model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("--judge_type", type=str, default="pointwise", choices=["pointwise", "pairwise", "pairwise_fine_grained", "alpaca_eval", "arena_hard_auto", "mt_bench", "mlr_bench"])
    parser.add_argument("--answer_position", type=str, default="first", choices=["first", "second"], help="The position of the answer in the pairwise comparison")
    parser.add_argument("--baseline_response_model_name", type=str, default=None, help="The model name of the baseline response model")
    parser.add_argument("--llm_agent_name", type=str, default="gpt-4.1-nano")
    parser.add_argument("--response_model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--dataset_name", type=str, default="AlpacaEval")
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--n_features", type=int, default=384)
    parser.add_argument("--test_mode", type=str, default="random", choices=["random", "single"])
    parser.add_argument("--data_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data")
    parser.add_argument("--eval_num", type=int, default=805)
    parser.add_argument("--reward_type", type=str, default="absolute", choices=["relative", "absolute"])
    parser.add_argument("--save_analysis_path", type=str, default="results/")
    parser.add_argument("--save_trajectory_path", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/")

    parser.add_argument("--alpha", type=float, default=1.0)

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # Run the async main function
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    finally:
        logger.info("Application finished.")