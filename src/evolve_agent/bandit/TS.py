import numpy as np
import random
import logging
import argparse
from tqdm import tqdm
import os
import json
import time

from src.evolve_agent.bandit.base import ContextualLinBanditAgent
from src.llm_evaluator import JudgeModel
from src.text_encoder import TextEncoder, MiniLMTextEncoder
from src.llm_zoo import ModelWrapper, load_model
from src.data.data_utils import load_dataset_for_exploration
from src.logging_utils import setup_logging
from src.data.assign_category import CATEGORIES

logger = logging.getLogger(__name__)

class ContextualLinThompsonSamplingAgent(ContextualLinBanditAgent):
    def __init__(self, n_features: int, llm_agent: ModelWrapper, embedding_model: TextEncoder, llm_evaluator: JudgeModel, reward_type: str = "relative", v_ts: float = 1.0, lambda_reg: float = 1.0):
        """
        Initializes the Contextual Thompson Sampling agent.

        Args:
            n_features (int): Dimension of context features (d).
            llm_agent (ModelWrapper): LLM agent to generate responses.
            embedding_model (TextEncoder): Model to get text embeddings.
            llm_evaluator (JudgeModel): LLM evaluator to evaluate the quality of the response.
            init_model_name (str): Name of the initial model being improved.
            reward_type (str): Type of reward to use. "relative" or "absolute".
            v_ts (float): Thompson Sampling exploration scale factor.
                          Controls the variance of the sampled parameters.
                          Higher v_ts means more exploration.
            lambda_reg (float): Regularization parameter for Ridge Regression.
                                This is the 'lambda' in (X^T X + lambda*I)^-1 X^T y.
                                Corresponds to initializing A_a with lambda_reg * I.
        """
        super().__init__(n_features, llm_agent, embedding_model, llm_evaluator, reward_type, lambda_reg)
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

if __name__ == "__main__":
    setup_logging(task_name="ThompsonSampling") # Changed task name for logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--Budget", type=int, default=20)
    parser.add_argument("--pool_size", type=int, default=3)
    parser.add_argument("--judge_model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("--llm_agent_name", type=str, default="gpt-4.1-nano")
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--n_features", type=int, default=384)
    parser.add_argument("--test_mode", type=str, default="single", choices=["random", "online", "single"])
    parser.add_argument("--dataset_name", type=str, default="AlpacaEval")
    parser.add_argument("--response_model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--data_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data") # Adjusted default path
    parser.add_argument("--eval_num", type=int, default=1000) # Reduced default for quicker testing
    parser.add_argument("--reward_type", type=str, default="relative", choices=["relative", "absolute"])
    parser.add_argument("--save_analysis_path", type=str, default="results/")
    parser.add_argument("--save_trajectory_path", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/trajectories/")

    parser.add_argument("--v_ts", type=float, default=1.0, help="Thompson Sampling exploration scale factor") # Added for TS

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    llm_agent = load_model(args.llm_agent_name)
    embedding_model = MiniLMTextEncoder()
    llm_evaluator = JudgeModel(args.judge_model_name)

    dataset = load_dataset_for_exploration(args.data_dir, args.dataset_name, args.response_model_name, args.judge_model_name)
    
    # preprocess the dataset, exclude the perfect score 9 samples
    test_results = []
    selected_idxs = []
    for idx in tqdm(range(len(dataset))):
        question, response, category, original_score, original_explanation = dataset[idx]['instruction'], dataset[idx]['output'], dataset[idx]['category'], dataset[idx]['original_score'], dataset[idx]['original_explanation']
        
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
                "skip": True,
            })
        elif original_score == -1:
            continue
        else:
            selected_idxs.append(idx)

    dataset_len = len(dataset)
    selected_idxs_len = len(selected_idxs)
    logger.info(f"Loaded {dataset_len} questions from {args.dataset_name}")
    logger.info(f"Selected {selected_idxs_len} valid questions from {dataset_len} questions")
    
    if args.eval_num >= selected_idxs_len:
        eval_num = selected_idxs_len
    else:
        eval_num = args.eval_num
        selected_idxs = random.sample(selected_idxs, eval_num)
        logger.info(f"Randomly sample {eval_num} questions from {selected_idxs_len} questions")
    logger.info("-"*100)

    # Exploration part
    logger.info(f"Skipped {len(test_results)} samples")
    logger.info(f"Dataset for exploration: {len(selected_idxs)} samples...")

    # selected samples
    dataset_for_exploration = [dataset[idx] for idx in selected_idxs]
    question_list = [item['instruction'] for item in dataset_for_exploration]
    init_response_list = [item['output'] for item in dataset_for_exploration]
    original_score_list = [item['original_score'] for item in dataset_for_exploration]
    original_explanation_list = [item['original_explanation'] for item in dataset_for_exploration]
    category_list = [item['category'] for item in dataset_for_exploration]

    logger.info(f"Initializing the agent...")
    agent = ContextualLinThompsonSamplingAgent(args.n_features, llm_agent, embedding_model, llm_evaluator, args.reward_type, args.v_ts, args.lambda_reg)
    logger.info(f"Agent initialized.")
    logger.info("-"*100)

    start_time = time.time()

    trajectories = []
    if args.test_mode == "online":
        logger.info(f"Running online learning...")
        trajectories = agent.online_learning(question_list, init_response_list, original_score_list, original_explanation_list, args.pool_size, args.Budget, shuffle_data=True, init_policy=True)
        logger.info(f"Online learning finished.")
        logger.info("-"*100)
    elif args.test_mode == "single":
        logger.info(f"Running single exploration...")
        trajectories = agent.batch_explore(question_list, init_response_list, original_score_list, original_explanation_list, args.pool_size, args.Budget, cold_start=True)
        logger.info(f"Single exploration finished.")
        logger.info("-"*100)
    elif args.test_mode == "random":
        logger.info(f"Running random exploration...")
        trajectories = agent.batch_explore_with_random_arm(question_list, init_response_list, original_score_list, original_explanation_list, args.pool_size, args.Budget)
        logger.info(f"Random exploration finished.")
        logger.info("-"*100)
    else:
        raise ValueError(f"Invalid test mode: {args.test_mode}")
    
    # keep on record the results
    for (question, response, category, original_score, original_explanation), trajectory in zip(zip(question_list, init_response_list, category_list, original_score_list, original_explanation_list), trajectories):
        final_score, final_explanation, final_response, _ = trajectory[-1]
        exploration_length = len(trajectory)-1
        test_results.append({
            "category": category,
            "instruction": question,
            "output": response,
            "original_score": original_score,
            "original_explanation": original_explanation,
            "final_score": final_score,
            "final_explanation": final_explanation,
            "final_response": final_response,
            "exploration_length": exploration_length,
            "skip": False,
        })
    
    end_time = time.time()
    
    # record the evaluation results
    analysis = {
        "strategy": "Thompson Sampling",
        "dataset_name": args.dataset_name,
        "response_model_name": args.response_model_name,
        "test_mode": args.test_mode,
        "judge_model_name": args.judge_model_name,
        "llm_agent_name": llm_agent.model_name,
        "lambda_reg": args.lambda_reg,
        "n_features": args.n_features,
        "budget": args.Budget,
        "pool_size": args.pool_size,
        "eval_num": eval_num,
        "reward_type": args.reward_type,
        "v_ts": args.v_ts,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "time_taken": end_time - start_time,
        "categories": {}
    }
    
    # Analyze the test result for each category
    for category in CATEGORIES:
        category_results = [result for result in test_results if result["category"] == category]
        logger.info(f"Category: {category}")
        logger.info(f"Number of results: {len(category_results)}")

        up_num = len([result for result in category_results if result["original_score"] < result["final_score"] and not result["skip"]])
        down_num = len([result for result in category_results if result["original_score"] > result["final_score"] and not result["skip"]])
        tie_num = len([result for result in category_results if result["original_score"] == result["final_score"] and not result["skip"]])
        skip_num = len([result for result in category_results if result["skip"]])
        avg_exploration_length = np.mean([result["exploration_length"] for result in category_results if not result["skip"]])
        avg_improvement = np.mean([result["final_score"] - result["original_score"] for result in category_results if not result["skip"]])
        avg_score_before = np.mean([result["original_score"] for result in category_results if not result["skip"]])
        avg_score_after = np.mean([result["final_score"] for result in category_results if not result["skip"]])
        
        logger.info(f"Number of up results: {up_num}")
        logger.info(f"Number of down results: {down_num}")
        logger.info(f"Number of tie results: {tie_num}")
        logger.info(f"Number of skip results: {skip_num}")
        logger.info(f"Average exploration length: {avg_exploration_length}")
        logger.info(f"Average improvement: {avg_improvement}")
        logger.info(f"Average score before: {avg_score_before}, average score after: {avg_score_after}")
        logger.info("--------------------------------")

        analysis["categories"][category] = {
            "up_num": up_num,
            "down_num": down_num,
            "tie_num": tie_num,
            "skip_num": skip_num,
            "exploration_length": avg_exploration_length,
            "average_improvement": avg_improvement,
            "avg_score_before": avg_score_before,
            "avg_score_after": avg_score_after,
        }
    
    # save the analysis in the output
    os.makedirs(args.save_analysis_path, exist_ok=True)
    save_path = os.path.join(args.save_analysis_path, f"evaluation_results.json")
    if os.path.exists(save_path):
        history_analysis = json.load(open(save_path, "r"))
        history_analysis.append(analysis)
    else:
        history_analysis = [analysis]
        
    with open(save_path, "w") as f:
        json.dump(history_analysis, f)
    logger.info(f"Analysis saved to {save_path}")
    logger.info("-"*100)

    # save the trajectories
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(args.save_trajectory_path, exist_ok=True)
    save_path = os.path.join(args.save_trajectory_path, f"ts_{args.test_mode}_{args.dataset_name}_{timestamp}.json")
    with open(save_path, "w") as f:
        json.dump(trajectories, f)
    logger.info(f"Trajectories saved to {save_path}")
    logger.info("-"*100)

    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds for exploration with {args.Budget} budget and {args.pool_size} pool size and {eval_num} eval num")