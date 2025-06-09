import numpy as np
import random
import heapq
import logging
import argparse
from tqdm import tqdm
import os
import json

from src.evolve_agent import EvolveAgent
from src.evolve_agent.bandit.config import strategy_list, STRATEGY_PROMPT
from src.llm_evaluator import JudgeModel
from src.text_encoder import TextEncoder, MiniLMTextEncoder
from src.llm_zoo import ModelWrapper, load_model
from src.utils import str2json
from src.data import load_dataset
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)

class ContextualThompsonSamplingAgent(EvolveAgent):
    def __init__(self, n_features: int, llm_agent: ModelWrapper, embedding_model: TextEncoder, llm_evaluator: JudgeModel, init_model_name: str, reward_type: str = "relative", v_ts: float = 1.0, lambda_reg: float = 1.0):
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
        super().__init__()

        self.n_arms = len(strategy_list) # number of arms
        self.strategy_list = strategy_list
        self.n_features = n_features
        self.v_ts = v_ts # Thompson Sampling parameter
        self.lambda_reg = lambda_reg
        self.llm_evaluator = llm_evaluator
        self.embedding_model = embedding_model
        self.llm_agent = llm_agent
        self.init_model_name = init_model_name
        self.reward_type = reward_type

        self.init_policy()

    def init_policy(self):
        # Initialize parameters for each arm
        # A_a: (d x d) matrix for each arm a. Stores (X_a^T X_a + lambda_reg * I)
        # b_a: (d x 1) vector for each arm a. Stores (X_a^T y_a)
        self.A = [np.identity(self.n_features) * self.lambda_reg for _ in range(self.n_arms)]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]

    def predict_sampled_rewards(self, context_x):
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

    def choose_arm(self, context_x):
        """
        Chooses an arm based on Thompson Sampling.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            int: The index of the chosen arm.
        """
        sampled_rewards = self.predict_sampled_rewards(context_x)
        chosen_arm_idx = np.argmax(sampled_rewards)
        return chosen_arm_idx

    def update(self, chosen_arm_idx, context_x, reward):
        """
        Updates the parameters for the chosen arm. (Identical to LinUCB)

        Args:
            chosen_arm_idx (int): The index of the arm that was played.
            context_x (np.array): The context vector (n_features x 1) for which the arm was played.
            reward (float): The observed reward.
        """
        if context_x.shape != (self.n_features, 1):
            raise ValueError(f"Context_x must be a column vector of shape ({self.n_features}, 1)")

        # Update A_a = A_a + x_t * x_t^T
        self.A[chosen_arm_idx] += context_x @ context_x.T # (d x 1) @ (1 x d) = (d x d)

        # Update b_a = b_a + r_t * x_t
        self.b[chosen_arm_idx] += reward * context_x # scalar * (d x 1) = (d x 1)

    def get_context_x(self, question: str, response: str):
        text = "Question: " + question + "\n" + "Response: " + response
        embedding = self.embedding_model.encode(text)
        return embedding # Should already be a numpy array

    def get_reward(self, question: str, response: str, original_score: float):
        s, e = self.llm_evaluator.pointwise_score(question, response)
        if self.reward_type == "relative":
            reward = s - original_score # Relative improvement as reward
        elif self.reward_type == "absolute":
            reward = s # Absolute score as reward
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
        return reward, s, e

    def get_policy_distribution(self, context_x):
        """
        For Thompson Sampling, this could represent the mean expected rewards
        before sampling, or one instance of sampled rewards.
        Let's provide the mean expected rewards for consistency with how one might
        interpret "policy distribution" as the current best guess without exploration noise.
        """
        policy_distribution = []
        for arm_idx in range(self.n_arms):
            A_a_inv = np.linalg.inv(self.A[arm_idx])
            theta_hat_a = A_a_inv @ self.b[arm_idx] # (d x d) @ (d x 1) = (d x 1)
            expected_reward = context_x.T @ theta_hat_a # (1 x d) @ (d x 1) = (1 x 1)
            policy_distribution.append(expected_reward.item()) # .item() to get scalar
        return policy_distribution

    def explore(self, question, init_response, pool_size: int, Budget: int):
        # init the response pool
        pool = []
        s, e = self.llm_evaluator.pointwise_score(question, init_response)
        original_init_score = s # Store the very first score
        pool.append((s, e, init_response))

        if Budget < self.n_arms:
            logger.warning(f"Budget ({Budget}) is less than the number of arms ({self.n_arms}). Initial forced exploration might not cover all arms.")
        
        # Initial forced exploration of each arm once using the initial response as context
        for arm_idx in range(min(self.n_arms, Budget)): # Ensure we don't exceed budget
            # Context for the first round can be based on the initial response
            # or a more general context if available. Here, using init_response.
            context_vector_np = self.get_context_x(question, init_response)
            context_x = context_vector_np.reshape(-1, 1) # Ensure (d,1)

            prompt = STRATEGY_PROMPT.format(question=question,
                                            response=init_response,
                                            N=1,
                                            strategy=self.strategy_list[arm_idx],
                                            feedback=e, # Feedback from initial response
                                            score=s,   # Score of initial response
                                            )
            new_response_list = self.llm_agent.invoke(prompt)
            # Ensure new_response_list is correctly parsed
            try:
                new_response = str2json(new_response_list)[0]
            except Exception as parse_ex:
                logger.error(f"Failed to parse LLM response for arm {self.strategy_list[arm_idx]}: {new_response_list}. Error: {parse_ex}. Skipping arm update.")
                continue # Skip this arm if parsing fails

            # Reward is relative to the score of the response *used to generate this new_response*
            # For the first round, this is the original_init_score
            reward, new_score, new_explanation = self.get_reward(question, new_response, original_init_score)
            self.update(arm_idx, context_x, reward) # Update with context of init_response

            # Add to pool based on absolute new_score
            heapq.heappush(pool, (new_score, new_explanation, new_response))
            if len(pool) > pool_size:
                heapq.heappop(pool)

            logger.info(f"Initial Pull - Iteration {arm_idx}:")
            logger.info(f"Original score (for context): {s}, explanation: {e}") # This 's' is from init_response
            logger.info(f"New score: {new_score}, explanation: {new_explanation}")
            logger.info(f"Chosen arm (forced): {self.strategy_list[arm_idx]}")
            logger.info(f"Reward calculated: {reward}")
            logger.info(f"New response: {new_response}\n")

        # Main Thompson Sampling loop
        # Start 't' from where initial forced exploration left off
        for t in range(min(self.n_arms, Budget), Budget):
            # 1. Sample a context (response) from the current pool
            # Use nlargest to get one of the best items as context, or random.choice for more diversity
            # For this example, let's pick a random one from the pool to avoid bias towards only high-scoring contexts
            if not pool:
                logger.warning("Pool is empty, cannot proceed with exploration.")
                break
            
            # Using random.choice from current pool members
            # heapq doesn't support random access, so convert to list first
            current_pool_list = list(pool)
            idx = random.choice(range(len(current_pool_list)))
            curr_s, curr_e, curr_r = current_pool_list[idx]

            context_vector_np = self.get_context_x(question, curr_r)
            context_x = context_vector_np.reshape(-1, 1)

            # 2. Choose an arm using Thompson Sampling
            chosen_arm = self.choose_arm(context_x)

            # 3. Get new response using the chosen arm and current context (curr_r, curr_s, curr_e)
            prompt = STRATEGY_PROMPT.format(question=question,
                                            response=curr_r,
                                            N=1,
                                            strategy=self.strategy_list[chosen_arm],
                                            feedback=curr_e,
                                            score=curr_s,
                                            )
            new_response_list = self.llm_agent.invoke(prompt)
            try:
                new_response = str2json(new_response_list)[0]
            except Exception as parse_ex:
                logger.error(f"Failed to parse LLM response for arm {self.strategy_list[chosen_arm]} in iter {t}: {new_response_list}. Error: {parse_ex}. Skipping update for this round.")
                continue

            # 4. Get the reward (relative to the score of curr_r)
            reward, new_score, new_explanation = self.get_reward(question, new_response, curr_s)

            # 5. Update the policy
            self.update(chosen_arm, context_x, reward)

            # 5.1 Update the pool with heapq
            heapq.heappush(pool, (new_score, new_explanation, new_response))
            if len(pool) > pool_size:
                heapq.heappop(pool)


            # 6. Log for evaluation
            logger.info(f"Iteration {t}:")
            logger.info(f"Context Response Score: {curr_s}, explanation: {curr_e}")
            logger.info(f"New Response Score: {new_score}, explanation: {new_explanation}")
            logger.info(f"Chosen arm (TS): {self.strategy_list[chosen_arm]}")
            logger.info(f"Reward calculated: {reward}")
            logger.info(f"New response: {new_response}\n")

        if not pool:
             logger.error("Pool is empty at the end of exploration. Returning initial response.")
             return original_init_score, e, init_response # Return original if pool became empty
        return heapq.nlargest(1, pool)[0] # Return the best from the pool

    def explore_with_random_arm(self, question, init_response, pool_size: int, Budget: int):
        # (This method remains largely the same, just ensure it uses the class structure correctly)
        pool = []
        s, e = self.llm_evaluator.pointwise_score(question, init_response)
        original_init_score = s # Store the very first score
        pool.append((s, e, init_response))

        for t in range(Budget):
            if not pool:
                logger.warning("Pool is empty in random exploration, cannot proceed.")
                break
            
            current_pool_list = list(pool)
            idx = random.choice(range(len(current_pool_list)))
            curr_s, curr_e, curr_r = current_pool_list[idx]

            chosen_arm = random.choice(range(self.n_arms))

            prompt = STRATEGY_PROMPT.format(question=question,
                                            response=curr_r,
                                            N=1,
                                            strategy=self.strategy_list[chosen_arm],
                                            feedback=curr_e,
                                            score=curr_s,
                                            )
            new_response_list = self.llm_agent.invoke(prompt)
            try:
                new_response = str2json(new_response_list)[0]
            except Exception as parse_ex:
                logger.error(f"Failed to parse LLM response for random arm {self.strategy_list[chosen_arm]} in iter {t}: {new_response_list}. Error: {parse_ex}. Skipping.")
                continue

            # Reward relative to curr_s
            reward, new_score, new_explanation = self.get_reward(question, new_response, curr_s)

            # For random exploration, we don't typically update the bandit model (A, b)
            # as it's a baseline. If you wanted to, you'd call self.update here.

            heapq.heappush(pool, (new_score, new_explanation, new_response))
            if len(pool) > pool_size:
                heapq.heappop(pool)

            logger.info(f"Random Iteration {t}:")
            logger.info(f"Context Response Score: {curr_s}, explanation: {curr_e}")
            logger.info(f"New Response Score: {new_score}, explanation: {new_explanation}")
            logger.info(f"Chosen arm (Random): {self.strategy_list[chosen_arm]}")
            logger.info(f"Reward calculated (not used for update): {reward}")
            logger.info(f"New response: {new_response}\n")
        
        if not pool:
             logger.error("Pool is empty at the end of random exploration. Returning initial response.")
             return original_init_score, e, init_response
        return heapq.nlargest(1, pool)[0]


categories = [
    "Computer Science & Programming",
    "Mathematics & Statistics",
    "Science & Engineering",
    "Business & Finance",
    "Writing & Communication",
    "Social & Daily Life",
    "Others"
]

if __name__ == "__main__":
    setup_logging(task_name="ThompsonSampling") # Changed task name for logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--Budget", type=int, default=15)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--judge_model_name", type=str, default="gemini-1.5-flash")
    parser.add_argument("--llm_agent_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--v_ts", type=float, default=1.0, help="Thompson Sampling exploration scale factor") # Added for TS
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--n_features", type=int, default=384)
    parser.add_argument("--save_path", type=str, default="results/")
    parser.add_argument("--test_mode", type=str, default="policy", choices=["random", "policy"])
    parser.add_argument("--dataset_name", type=str, default="AlpacaEval")
    parser.add_argument("--response_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--data_dir", type=str, default="/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data") # Adjusted default path
    parser.add_argument("--eval_num", type=int, default=10) # Reduced default for quicker testing
    parser.add_argument("--reward_type", type=str, default="relative", choices=["relative", "absolute"])

    args = parser.parse_args()

    dataset_name = args.dataset_name
    response_model_name = args.response_model_name

    n_features = args.n_features
    llm_agent = load_model(args.llm_agent_name)
    embedding_model = MiniLMTextEncoder()
    judge_model_name = args.judge_model_name
    llm_evaluator = JudgeModel(judge_model_name)
    v_ts = args.v_ts # Use v_ts
    lambda_reg = args.lambda_reg
    eval_num = args.eval_num
    reward_type = args.reward_type

    budget = args.Budget
    pool_size = args.pool_size
    data_dir = args.data_dir

    logger.info(f"Using Thompson Sampling Agent")
    logger.info(f"Dataset: {dataset_name}, Response model: {response_model_name}")
    logger.info(f"Budget: {budget}, Pool size: {pool_size}")
    logger.info(f"v_ts (TS exploration): {v_ts}, Lambda reg: {lambda_reg}")
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
        dataset = random.sample(list(dataset), eval_num) # Ensure dataset is a list for random.sample
        logger.info(f"Randomly sample {eval_num} questions from {dataset_len} questions")
    logger.info("-"*100)


    test_results = []
    for idx in tqdm(range(len(dataset))):
        # Ensure dataset item has expected keys
        item = dataset[idx]
        if not all(k in item for k in ['instruction', 'output', 'category']):
            logger.error(f"Dataset item {idx} is missing required keys: {item}. Skipping.")
            continue

        question, response, category = item['instruction'], item['output'], item['category']
        logger.info(f"Question {idx}: {question}")
        logger.info(f"Initial Response {idx}: {response}")

        # Instantiate Thompson Sampling Agent
        agent = ContextualThompsonSamplingAgent(n_features, llm_agent, embedding_model, llm_evaluator, response_model_name, reward_type, v_ts, lambda_reg)

        original_score, original_explanation = llm_evaluator.pointwise_score(question, response)
        logger.info(f"Original score: {original_score}, explanation: {original_explanation}")
        if original_score is None: # Handle evaluator failure
            logger.error(f"LLM Evaluator failed for initial response of question {idx}. Skipping.")
            test_results.append({
                "category": category, "instruction": question, "output": response,
                "original_score": None, "original_explanation": "Evaluator Failed",
                "final_score": None, "final_explanation": "Evaluator Failed",
                "final_response": response, "skip": True, "error": "Initial Eval Failed"
            })
            continue

        if original_score >= 9: # Assuming 9 is a near-perfect score
            logger.info(f"High original score {original_score}, skipping bandit exploration.")
            test_results.append({
                "category": category, "instruction": question, "output": response,
                "original_score": original_score, "original_explanation": original_explanation,
                "final_score": original_score, "final_explanation": original_explanation,
                "final_response": response, "skip": True,
            })
            continue
        
        final_score, final_explanation, final_response = -1, "Error", response # Default values in case of error
        try:
            if args.test_mode == "random":
                final_score, final_explanation, final_response = agent.explore_with_random_arm(question, response, pool_size, budget)
            elif args.test_mode == "policy":
                final_score, final_explanation, final_response = agent.explore(question, response, pool_size, budget)
            else:
                raise ValueError(f"Invalid test mode: {args.test_mode}")
        except Exception as e_explore:
            logger.error(f"Error during agent.explore for question {idx}: {e_explore}", exc_info=True)
            # test_results will use default error values

        logger.info(f"Final score: {final_score}, explanation: {final_explanation}")
        logger.info(f"Final response: {final_response}")
        logger.info("-"*100)

        test_results.append({
            "category": category, "instruction": question, "output": response,
            "original_score": original_score, "original_explanation": original_explanation,
            "final_score": final_score, "final_explanation": final_explanation,
            "final_response": final_response, "skip": False,
        })

    # Analysis part (similar to original, adjust strategy name)
    analysis = {
        "strategy": "ThompsonSampling", # Changed strategy name
        "dataset_name": dataset_name,
        "response_model_name": response_model_name,
        "test_mode": args.test_mode,
        "judge_model_name": judge_model_name,
        "llm_agent_name": llm_agent.model_name,
        "v_ts": v_ts, # Store v_ts
        "lambda_reg": lambda_reg,
        "n_features": n_features,
        "budget": budget,
        "pool_size": pool_size,
        "eval_num": eval_num,
        "reward_type": reward_type,
    }

    # Filter out results where scores might be None due to errors
    valid_results_for_analysis = [r for r in test_results if r.get("original_score") is not None and r.get("final_score") is not None and isinstance(r.get("original_score"), (int, float)) and isinstance(r.get("final_score"), (int, float))]

    for category_name in categories: # Ensure using correct variable name
        category_results = [result for result in valid_results_for_analysis if result["category"] == category_name]
        logger.info(f"Category: {category_name}") # Corrected variable name
        if not category_results:
            logger.info("No valid results for this category.")
            avg_improvement = 0
            up_num, down_num, tie_num = 0,0,0
        else:
            logger.info(f"Number of valid results: {len(category_results)}")
            up_num = len([result for result in category_results if result["original_score"] < result["final_score"] and not result.get("skip", False)])
            down_num = len([result for result in category_results if result["original_score"] > result["final_score"] and not result.get("skip", False)])
            tie_num = len([result for result in category_results if result["original_score"] == result["final_score"] and not result.get("skip", False)])
            avg_improvement_list = [result['final_score'] - result['original_score'] for result in category_results if not result.get("skip", False)]
            avg_improvement = np.mean(avg_improvement_list) if avg_improvement_list else 0

        logger.info(f"Number of up results: {up_num}")
        logger.info(f"Number of down results: {down_num}")
        logger.info(f"Number of tie results: {tie_num}")
        # Count skips from original test_results, not just valid_results_for_analysis if skip means something different
        total_category_results = [r for r in test_results if r["category"] == category_name]
        skip_num_total = len([result for result in total_category_results if result.get("skip", False)])
        logger.info(f"Number of skip results (total): {skip_num_total}")
        logger.info(f"Average improvement (on non-skipped, valid): {avg_improvement}")
        logger.info("--------------------------------")

        analysis[category_name] = {
            "up_num": up_num,
            "down_num": down_num,
            "tie_num": tie_num,
            "skip_num": skip_num_total,
            "average_improvement": float(avg_improvement), # Ensure JSON serializable
        }

    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)
    results_filename = f"TS_results_{dataset_name}_{response_model_name}_{args.test_mode}_B{budget}_VTS{v_ts}.json"
    analysis_filename = f"TS_analysis_{dataset_name}_{response_model_name}_{args.test_mode}_B{budget}_VTS{v_ts}.json"
    
    full_results_save_path = os.path.join(save_dir, results_filename)
    with open(full_results_save_path, "w") as f:
        json.dump(test_results, f, indent=4)
    logger.info(f"Full test results saved to {full_results_save_path}")

    # For aggregated analysis file (appending):
    aggregated_analysis_file = os.path.join(save_dir, "evaluation_summary.json")
    try:
        if os.path.exists(aggregated_analysis_file):
            with open(aggregated_analysis_file, "r") as f:
                history_analysis = json.load(f)
                if not isinstance(history_analysis, list): # handle case where file is not a list
                    history_analysis = [history_analysis]
        else:
            history_analysis = []
        history_analysis.append(analysis)
        with open(aggregated_analysis_file, "w") as f:
            json.dump(history_analysis, f, indent=4)
        logger.info(f"Aggregated analysis saved/appended to {aggregated_analysis_file}")
    except json.JSONDecodeError:
        logger.error(f"Error reading {aggregated_analysis_file}. Saving current analysis as a new list.")
        with open(aggregated_analysis_file, "w") as f:
            json.dump([analysis], f, indent=4)