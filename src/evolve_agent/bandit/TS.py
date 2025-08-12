import numpy as np
import logging
import argparse
import time
import asyncio

from src.evolve_agent.bandit.base import ContextualLinBanditAgent
from src.llm_evaluator import JudgeType
from src.text_encoder import TextEncoder, MiniLMTextEncoder
from src.llm_zoo import BaseLLM, load_model
from src.logging_utils import setup_logging
from src.evolve_agent.utils import prepare_dataset_for_exploration, exclude_perfect_response, sample_and_filter_data, extract_result_from_trajectories, get_result_analysis, save_result_analysis, save_trajectories
from src.llm_zoo.api_zoo import get_model_name

logger = logging.getLogger(__name__)

class ContextualLinThompsonSamplingAgent(ContextualLinBanditAgent):
    def __init__(self, n_features: int, llm_agent: BaseLLM, embedding_model: TextEncoder, judge_type: JudgeType, judge_model_backbone: str, reward_type: str = "relative", v_ts: float = 1.0, lambda_reg: float = 1.0, answer_position: str = "first"):
        """
        Initializes the Contextual Thompson Sampling agent.

        Args:
            n_features (int): Dimension of context features (d).
            llm_agent (BaseLLM): LLM agent to generate responses.
            embedding_model (TextEncoder): Model to get text embeddings.
            judge_type (JudgeType): Type of judge evaluation (pointwise, pairwise, etc.)
            judge_model_backbone (str): Backbone model for the judge
            reward_type (str): Type of reward to use ("relative" or "absolute").
            v_ts (float): Thompson Sampling exploration scale factor.
                          Controls the variance of the sampled parameters.
                          Higher v_ts means more exploration.
            lambda_reg (float): Regularization parameter for Ridge Regression.
                                This is the 'lambda' in (X^T X + lambda*I)^-1 X^T y.
                                Corresponds to initializing A_a with lambda_reg * I.
            answer_position (str): Position of the answer in pairwise comparison ("first" or "second").
        """
        super().__init__(n_features, llm_agent, embedding_model, judge_type, judge_model_backbone, reward_type, lambda_reg, answer_position)
        self.v_ts = v_ts # Thompson Sampling parameter
        self.init_policy_model()

    def predict(self, context_x):
        """
        Predicts the sampled expected reward for each arm given the context using Thompson Sampling.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            np.array: A (n_arms x 1) vector of sampled expected rewards for each arm.
        """
        if context_x.shape != (self.n_features, 1):
            raise ValueError(f"Context_x must be a column vector of shape ({self.n_features}, 1)")

        sampled_rewards = np.zeros((self.n_arms, 1))

        for arm_idx in range(self.n_arms):
            A_a_inv = np.linalg.inv(self.A[arm_idx])
            theta_hat_a = A_a_inv @ self.b[arm_idx] # Mean of the posterior (d x 1)

            # Covariance of the posterior for theta_a is v_ts^2 * A_a_inv
            # Ensure theta_hat_a is 1D for multivariate_normal
            mean_theta_hat_a = theta_hat_a.flatten()
            
            # Sample theta_tilde_a from N(theta_hat_a, v_ts^2 * A_a_inv)
            # Add a small epsilon to diagonal for numerical stability if needed, but usually fine
            try:
                # Ensure covariance matrix is positive semi-definite. Sometimes numerical issues occur.
                # A common fix is to add a small identity matrix, or ensure A_a_inv is symmetric.
                # A_a_inv should be symmetric if A_a is symmetric.
                cov_matrix = self.v_ts**2 * A_a_inv
                # Symmetrize to be sure, helps with numerical stability for np.random.multivariate_normal
                cov_matrix = (cov_matrix + cov_matrix.T) / 2.0

                theta_tilde_a = np.random.multivariate_normal(mean_theta_hat_a, cov_matrix)
                theta_tilde_a = theta_tilde_a.reshape(-1, 1) # Reshape back to (d x 1)
            except np.linalg.LinAlgError as e:
                # Fallback if covariance matrix is not positive semi-definite
                # This can happen early on or with unstable data.
                # A simple fallback is to use the mean, or add more regularization.
                logger.warning(f"LinAlgError sampling for arm {arm_idx}: {e}. Using mean theta_hat_a.")
                theta_tilde_a = theta_hat_a

            # Expected reward using the sampled theta_tilde_a: x^T * theta_tilde_a
            sampled_rewards[arm_idx] = context_x.T @ theta_tilde_a # (1 x d) @ (d x 1) = (1 x 1)

        return sampled_rewards
    
    def batch_predict(self, context_x_list):
        """
        Predicts the sampled expected reward for each arm given the context using Thompson Sampling.
        Args:
            context_x_list (np.array): A (n_samples, n_features, 1) tensor representing the context.
        Returns:
            np.array: A (n_samples, n_arms, 1) tensor of sampled expected rewards for each arm.
        """
        if context_x_list.shape[1:] != (self.n_features, 1):
            raise ValueError(f"Context_x must be a tensor of shape (n_samples, {self.n_features}, 1)")
        
        n_samples = context_x_list.shape[0]
        sampled_rewards = np.zeros((n_samples, self.n_arms, 1))
        
        for arm_idx in range(self.n_arms):
            A_a_inv = np.linalg.inv(self.A[arm_idx])
            theta_hat_a = A_a_inv @ self.b[arm_idx]  # Mean of the posterior (d x 1)
            mean_theta_hat_a = theta_hat_a.flatten()
            
            # Covariance of the posterior for theta_a is v_ts^2 * A_a_inv
            cov_matrix = self.v_ts**2 * A_a_inv
            # Symmetrize to be sure, helps with numerical stability
            cov_matrix = (cov_matrix + cov_matrix.T) / 2.0
            
            try:
                theta_tilde_a = np.random.multivariate_normal(mean_theta_hat_a, cov_matrix)
                theta_tilde_a = theta_tilde_a.reshape(-1, 1)  # Reshape back to (d x 1)
            except np.linalg.LinAlgError as e:
                logger.warning(f"LinAlgError sampling for arm {arm_idx}: {e}. Using mean theta_hat_a.")
                theta_tilde_a = theta_hat_a
            
            # Reshape context_x_list for batch operations
            context_x_reshaped = context_x_list.reshape(n_samples, self.n_features)  # (n_samples, n_features)
            
            # Expected reward using the sampled theta_tilde_a: x^T * theta_tilde_a
            sampled_rewards[:, arm_idx] = (context_x_reshaped @ theta_tilde_a).reshape(-1, 1)  # (n_samples, 1)
            
        return sampled_rewards

    def choose_arm(self, context_x):
        """
        Chooses an arm based on Thompson Sampling.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            int: The index of the chosen arm.
        """
        sampled_rewards = self.predict(context_x)
        chosen_arm_idx = np.argmax(sampled_rewards)
        return chosen_arm_idx
    
    def batch_choose_arm(self, context_x_list):
        """
        Chooses an arm based on Thompson Sampling.
        Args:
            context_x_list (np.array): A (n_samples, n_features, 1) tensor representing the context.
        Returns:
            np.array: A (n_samples,) tensor of the chosen arm index for each sample.
        """
        sampled_rewards = self.batch_predict(context_x_list)
        chosen_arm_idxs = np.argmax(sampled_rewards, axis=1)
        return chosen_arm_idxs

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
    agent = ContextualLinThompsonSamplingAgent(args.n_features, llm_agent, embedding_model, judge_type, judge_model_backbone, args.reward_type, args.v_ts, args.lambda_reg, args.answer_position)
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
        "strategy": "Thompson Sampling",
        "judge_type": args.judge_type,
        "answer_position": args.answer_position,
        "dataset_name": args.dataset_name,
        "judge_backbone": get_model_name(args.judge_model_name),
        "baseline_response_model_name": get_model_name(args.baseline_response_model_name),
        "llm_agent_name": get_model_name(args.llm_agent_name),
        "response_model_name": get_model_name(args.response_model_name),
        "test_mode": args.test_mode,
        "lambda_reg": args.lambda_reg,
        "n_features": args.n_features,
        "budget": args.Budget,
        "pool_size": args.pool_size,
        "eval_num": eval_num,
        "reward_type": args.reward_type,
        "v_ts": args.v_ts,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_taken": end_time - start_time,
    }
    analysis.update(meta_info)
    trajectories_to_save = meta_info.copy()
    trajectories_to_save.update({"trajectories": trajectories})

    save_result_analysis(analysis, args.save_analysis_path)
    save_trajectories(trajectories_to_save, args.save_trajectory_path, f"ts")

    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds for exploration with {args.Budget} budget and {args.pool_size} pool size and {eval_num} eval num")


if __name__ == "__main__":
    setup_logging(task_name="ThompsonSampling")

    parser = argparse.ArgumentParser()
    parser.add_argument("--Budget", type=int, default=3)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--judge_model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("--judge_type", type=str, default="pairwise", choices=["pointwise", "pairwise", "pairwise_fine_grained", "alpaca_eval", "arena_hard_auto", "mt_bench", "mlr_bench"])
    parser.add_argument("--answer_position", type=str, default="first", choices=["first", "second"], help="The position of the answer in the pairwise comparison")
    parser.add_argument("--baseline_response_model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--llm_agent_name", type=str, default="gpt-4.1-nano")
    parser.add_argument("--response_model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--dataset_name", type=str, default="AlpacaEval")
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--n_features", type=int, default=384)
    parser.add_argument("--test_mode", type=str, default="random", choices=["random", "single"])
    parser.add_argument("--data_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data")
    parser.add_argument("--eval_num", type=int, default=2)
    parser.add_argument("--reward_type", type=str, default="relative", choices=["relative", "absolute"])
    parser.add_argument("--save_analysis_path", type=str, default="results/")
    parser.add_argument("--save_trajectory_path", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/")

    parser.add_argument("--v_ts", type=float, default=1.0, help="Thompson Sampling exploration scale factor")

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # Run the async main function
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    finally:
        logger.info("Application finished.")