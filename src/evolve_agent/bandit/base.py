'''Base class for Contextual Bandit Agents.

TODO: Can try Neural Bandit in the future.
'''
import numpy as np
import random
import heapq
import logging
from tqdm import tqdm
from abc import abstractmethod
from sklearn.utils import shuffle

from src.evolve_agent import EvolveAgent
from src.llm_zoo import ModelWrapper
from src.text_encoder import TextEncoder
from src.llm_evaluator import JudgeModel
from src.evolve_agent.utils import find_shortest_of_max_simple
from src.evolve_agent.bias_strategies import Bias_types, BiasModification
from src.evolve_agent.bandit.reward_cal import PointwiseRewardCalculator, PairwiseRewardCalculator

logger = logging.getLogger(__name__)

class ContextualBanditAgent(EvolveAgent):
    def __init__(self, n_features: int, llm_agent: ModelWrapper, embedding_model: TextEncoder, llm_evaluator: JudgeModel, reward_type: str = "relative", judge_type: str = "pointwise"):
        """
        Initializes the Bandit Agent.

        Args:
            n_features (int): Dimension of context features (d).
            llm_agent (ModelWrapper): LLM agent to generate the response. 
            embedding_model (TextEncoder): Embedding model to encode the context.
            llm_evaluator (JudgeModel): LLM evaluator to evaluate the quality of the response.
            reward_type (str): Type of reward to use.
        """
        super().__init__()
        self.n_arms = len(Bias_types) # number of arms
        self.strategy_list = Bias_types
        self.n_features = n_features
        self.llm_evaluator = llm_evaluator
        self.embedding_model = embedding_model
        self.llm_agent = llm_agent
        self.reward_type = reward_type
        self.judge_type = judge_type
        self.bias_modificatior = BiasModification(llm_agent)
        self.reward_fn = PointwiseRewardCalculator(llm_evaluator, reward_type) if judge_type == "pointwise" else PairwiseRewardCalculator(llm_evaluator, reward_type)
    
    @abstractmethod
    def init_policy_model(self):
        pass

    @abstractmethod
    def predict(self, context_x):
        """
        Predicts the score for each arm given the context.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            np.array: A (n_arms x 1) vector of UCB scores for each arm.
        """
        pass
    
    @abstractmethod
    def choose_arm(self, context_x):
        """
        Chooses an arm based on the specific strategy.

        Args:
            context_x (np.array): A (n_features x 1) column vector representing the context.

        Returns:
            int: The index of the chosen arm.
        """
        pass

    @abstractmethod
    def batch_predict(self, context_x_list):
        """
        Predicts the score for each arm given the context.
        """
        pass

    @abstractmethod
    def batch_choose_arm(self, context_x_list):
        """
        Chooses an arm based on the specific strategy.
        """
        pass

    @abstractmethod
    def update(self, chosen_arm_idx, context_x, reward):
        """
        Updates the parameters for the chosen arm.

        Args:
            chosen_arm_idx (int): The index of the arm that was played.
            context_x (np.array): The context vector (n_features x 1) for which the arm was played.
            reward (float): The observed reward.
        """
        pass
        
    def get_context_x(self, question: str, response: str):
        text = "Question: " + question + "\n" + "Response: " + response
        embedding = self.embedding_model.encode(text)
        return embedding
    
    def get_context_x_batch(self, question_list: list[str], response_list: list[str]):
        texts = ["Question: " + question + "\n" + "Response: " + response for question, response in zip(question_list, response_list)]
        embeddings = self.embedding_model.batch_encode(texts)
        # shape: (n_samples, n_features)
        return embeddings
    
    def get_reward(self, question: str, response: str, original_score: float, baseline_response: str = None):
        if self.judge_type == "pointwise":
            return self.reward_fn.calculate_reward(question, response, original_score)
        elif self.judge_type == "pairwise":
            return self.reward_fn.calculate_reward(question, response, original_score, baseline_response)
        else:
            raise ValueError(f"Invalid judge type: {self.judge_type}")
    
    def get_batch_reward(self, question_list: list[str], response_list: list[str], original_score_list: list[float], baseline_response_list: list[str] = None):
        if self.judge_type == "pointwise":
            return self.reward_fn.calculate_batch_reward(question_list, response_list, original_score_list)
        elif self.judge_type == "pairwise":
            return self.reward_fn.calculate_batch_reward(question_list, response_list, original_score_list, baseline_response_list)
        else:
            raise ValueError(f"Invalid judge type: {self.judge_type}")
    
    def explore(self, question, init_response, original_score, original_explanation, pool_size: int, Budget: int, cold_start: bool, baseline_response: str = None):
        """
        Explore responses using contextual bandit algorithm.
        
        Args:
            question (str): The input question
            init_response (str): Initial response to start with
            original_score (float): Original score of the initial response
            original_explanation (str): Explanation for the original score
            pool_size (int): Maximum size of the response pool
            Budget (int): Number of exploration steps
            cold_start (bool): Whether to perform cold start exploration
            baseline_response (str, optional): Pre-calculated baseline response for pairwise comparison
        """
        if Budget<self.n_arms:
            raise ValueError(f"Budget must be greater than the number of arms: {self.n_arms}")
        # init the response pool
        pool = []
        pool.append([original_score, (original_score, original_explanation, init_response, None)])

        curr_step = 0
        if cold_start:
            curr_path = pool[0].copy()
            curr_s, curr_e, curr_r, _ = curr_path[-1]
            context_x = self.get_context_x(question, init_response)
            context_x = context_x.reshape(-1, 1)
            for arm_idx in range(self.n_arms):
        
                strategy = Bias_types[arm_idx]

                new_response = self.bias_modificatior.principle_guided_mutation(init_response, strategy)

                # Use pre-calculated baseline response for pairwise evaluation
                reward, new_score, new_explanation = self.get_reward(question, new_response, curr_s, baseline_response)
                self.update(arm_idx, context_x, reward)

                pool.append([new_score, (original_score, original_explanation, init_response, None), (new_score, new_explanation, new_response, strategy)])
                if len(pool) > pool_size:
                    heapq.heappop(pool)

                # record the results
                logger.info(f"Iteration {arm_idx}:")
                logger.info(f"Original score: {original_score}, explanation: {original_explanation}")
                logger.info(f"New score: {new_score}, explanation: {new_explanation}")
                logger.info(f"Chosen arm: {self.strategy_list[arm_idx]}")
                logger.info(f"New response: {new_response}\n")
                curr_step += 1
        
        for t in range(curr_step, Budget):
            # 1. Sample a context from the dataset
            idx = random.choice(range(len(pool)))
            
            curr_path = pool[idx].copy()
            curr_s, curr_e, curr_r, _ = curr_path[-1]

            context_x = self.get_context_x(question, curr_r)
            context_x = context_x.reshape(-1, 1)

            # 2. Choose an arm
            chosen_arm = self.choose_arm(context_x)

            # 3. get new response
            strategy = Bias_types[chosen_arm]
            new_response = self.bias_modificatior.principle_guided_mutation(curr_r, strategy)

            # 4. Get the reward
            # Use pre-calculated baseline response for pairwise evaluation
            reward, new_score, new_explanation = self.get_reward(question, new_response, curr_s, baseline_response)

            # 5. Update the policy
            self.update(chosen_arm, context_x, reward)

            # 5.1 update the pool with heapq
            curr_path.append((new_score, new_explanation, new_response, strategy))
            curr_path[0] = new_score
            pool.append(curr_path.copy())
            if len(pool) > pool_size:
                heapq.heappop(pool)
            
            # 6. log for evaluation
            logger.info(f"Iteration {t}:")
            logger.info(f"Original score: {curr_s}, explanation: {curr_e}")
            logger.info(f"New score: {new_score}, explanation: {new_explanation}")
            logger.info(f"Chosen arm: {self.strategy_list[chosen_arm]}")
            logger.info(f"New response: {new_response}\n")

        best_path = find_shortest_of_max_simple(pool)
        return best_path
    
    def batch_explore(self, question_list, init_response_list, original_score_list, original_explanation_list, pool_size: int, Budget: int, cold_start: bool, baseline_response_list: list[str] = None):
        """
        Batch explore responses using contextual bandit algorithm.
        
        Args:
            question_list (list[str]): List of input questions
            init_response_list (list[str]): List of initial responses
            original_score_list (list[float]): List of original scores
            original_explanation_list (list[str]): List of original explanations
            pool_size (int): Maximum size of the response pool
            Budget (int): Number of exploration steps
            cold_start (bool): Whether to perform cold start exploration
            baseline_response_list (list[str], optional): List of pre-calculated baseline responses for pairwise comparison
        """
        if Budget<self.n_arms:
            raise ValueError(f"Budget must be greater than the number of arms: {self.n_arms}")
        
        pool_list = []
        for init_response, original_score, original_explanation in zip(init_response_list, original_score_list, original_explanation_list):
            pool = []
            pool.append([original_score, (original_score, original_explanation, init_response, None)])
            pool_list.append(pool.copy())

        curr_step = 0
        if cold_start:
            # shape: (n_samples, n_features)
            context_x_list = self.get_context_x_batch(question_list, init_response_list)
            context_x_list = context_x_list.reshape(len(init_response_list), self.n_features, 1)
            for arm_idx in range(self.n_arms):
                strategy_list = [Bias_types[arm_idx]] * len(init_response_list)
                new_responses = self.bias_modificatior.batch_principle_guided_mutation(init_response_list, strategy_list)
                reward_list, new_score_list, new_explanation_list = self.get_batch_reward(question_list, new_responses, original_score_list, baseline_response_list)

                for i in range(len(init_response_list)):
                    self.update(arm_idx, context_x_list[i], reward_list[i])
                
                for pool, new_s, new_e, new_r, new_strategy, original_score, original_explanation, init_response in zip(pool_list, new_score_list, new_explanation_list, new_responses, strategy_list, original_score_list, original_explanation_list, init_response_list):
                    pool.append([new_s, (original_score, original_explanation, init_response, None), (new_s, new_e, new_r, new_strategy)])
                    if len(pool) > pool_size:
                        heapq.heappop(pool)
                
                curr_step += 1

        for t in range(curr_step, Budget):
            # 1. Sample a context from the dataset
            response_list = []
            sampled_path_list = []
            for pool in pool_list:
                idx = random.choice(range(len(pool)))
                curr_path = pool[idx].copy()
                sampled_path_list.append(curr_path)
                response_list.append(curr_path[-1][2])

            context_x_list = self.get_context_x_batch(question_list, response_list)
            context_x_list = context_x_list.reshape(len(response_list), self.n_features, 1)

            # 2. Choose an arm, shape: (n_samples, 1)
            chosen_arm_list = self.batch_choose_arm(context_x_list)

            # 3. get new response
            strategy_list = [Bias_types[chosen_arm[0]] for chosen_arm in chosen_arm_list]
            new_response_list = self.bias_modificatior.batch_principle_guided_mutation(response_list, strategy_list)
            
            # 4. Get the reward
            reward_list, new_score_list, new_explanation_list = self.get_batch_reward(question_list, new_response_list, original_score_list, baseline_response_list)

            # 5. Update the policy
            for i in range(len(response_list)):
                self.update(chosen_arm_list[i][0], context_x_list[i], reward_list[i])

            # 5.1 update the pool with heapq
            for curr_path, new_s, new_e, new_r, new_strategy, pool in zip(sampled_path_list, new_score_list, new_explanation_list, new_response_list, strategy_list, pool_list):
                curr_path.append((new_s, new_e, new_r, new_strategy))
                curr_path[0] = new_s
                pool.append(curr_path.copy())
                if len(pool) > pool_size:
                    heapq.heappop(pool)
        
        # 6. find the best path for each question
        best_path_list = []
        for pool in pool_list:
            best_path = find_shortest_of_max_simple(pool)
            best_path_list.append(best_path.copy())
        return best_path_list
    
    def explore_with_random_arm(self, question, init_response, original_score, original_explanation, pool_size: int, Budget: int):
        # init the response pool
        pool = []
        pool.append([original_score, (original_score, original_explanation, init_response, None)])
        
        for t in range(Budget):
            # 1. Sample a context from the dataset
            idx = random.choice(range(len(pool)))
            curr_path = pool[idx].copy()
            curr_s, curr_e, curr_r, _ = curr_path[-1]

            # 2. Choose an arm
            chosen_arm = random.choice(range(self.n_arms))

            # 3. get new response
            strategy = Bias_types[chosen_arm]
            new_response = self.bias_modificatior.principle_guided_mutation(curr_r, strategy)

            # 4. Get the reward
            new_score, new_explanation = self.llm_evaluator.pointwise_score(question, new_response)

            # 5.1 update the pool with heapq
            curr_path.append((new_score, new_explanation, new_response, strategy))
            curr_path[0] = new_score
            pool.append(curr_path.copy())
            
            if len(pool) > pool_size:
                heapq.heappop(pool)
            
            # 6. log for evaluation
            print(f"Iteration {t}:")
            print(f"Original score: {curr_s}, explanation: {curr_e}")
            print(f"New score: {new_score}, explanation: {new_explanation}")
            print(f"Chosen arm: {self.strategy_list[chosen_arm]}")
            print(f"New response: {new_response}\n")
        
        # return the best response with shortest length
        best_path = find_shortest_of_max_simple(pool)
        return best_path
    
    def batch_explore_with_random_arm(self, question_list, init_response_list, original_score_list, original_explanation_list, pool_size: int, Budget: int):
        pool_list = []
        for init_response, original_score, original_explanation in zip(init_response_list, original_score_list, original_explanation_list):
            pool = []
            pool.append([original_score, (original_score, original_explanation, init_response, None)])
            pool_list.append(pool.copy())

        for t in range(Budget):
            # 1. Sample a context from the dataset
            response_list = []
            sampled_path_list = []
            for pool in pool_list:
                idx = random.choice(range(len(pool)))
                curr_path = pool[idx].copy()
                sampled_path_list.append(curr_path)
                response_list.append(curr_path[-1][2])

            # 2. Choose an arm
            chosen_arm_list = [random.choice(range(self.n_arms))] * len(response_list)

            # 3. get new response
            strategy_list = [Bias_types[chosen_arm] for chosen_arm in chosen_arm_list]
            new_response_list = self.bias_modificatior.batch_principle_guided_mutation(response_list, strategy_list)
            
            # 4. Get the score
            new_score_list, new_explanation_list = self.llm_evaluator.batch_pointwise_score(question_list, new_response_list)

            # 5.1 update the pool with heapq
            for curr_path, new_s, new_e, new_r, new_strategy, pool in zip(sampled_path_list, new_score_list, new_explanation_list, new_response_list, strategy_list, pool_list):
                curr_path.append((new_s, new_e, new_r, new_strategy))
                curr_path[0] = new_s
                pool.append(curr_path.copy())
                if len(pool) > pool_size:
                    heapq.heappop(pool)
        
        # 6. find the best path for each question
        best_path_list = []
        for pool in pool_list:
            best_path = find_shortest_of_max_simple(pool)
            best_path_list.append(best_path.copy())
        return best_path_list

    
    def online_learning(self, question_list, init_response_list, original_score_list, original_explanation_list, pool_size: int, Budget: int, shuffle_data: bool = True, init_policy: bool = True):
        # shuffle the question and init_response (Optional)
        if shuffle_data:
            logger.info(f"Shuffling the question and init_response")
            question_list, init_response_list, original_score_list, original_explanation_list = shuffle(question_list, init_response_list, original_score_list, original_explanation_list)
        
        if init_policy:
            logger.info(f"Initializing the policy model")
            self.init_policy_model()
        
        explore_trajectories = []
        # online learning
        logger.info(f"Online learning started")
        for t, (question, init_response, original_score, original_explanation) in tqdm(enumerate(zip(question_list, init_response_list, original_score_list, original_explanation_list))):
            explore_trajectory = self.explore(question, init_response, original_score, original_explanation, pool_size, Budget, cold_start=True if t==0 else False)
            explore_trajectories.append(explore_trajectory)
            logger.info(f"Online learning iteration {t} finished")
            logger.info("-"*100)

        logger.info(f"Online learning finished")
        return explore_trajectories
    

class ContextualLinBanditAgent(ContextualBanditAgent):
    def __init__(self, n_features: int, llm_agent: ModelWrapper, embedding_model: TextEncoder, llm_evaluator: JudgeModel, reward_type: str = "relative", lambda_reg: float = 1.0):
        super().__init__(n_features, llm_agent, embedding_model, llm_evaluator, reward_type)
        self.lambda_reg = lambda_reg

    def init_policy_model(self):
        # Initialize parameters for each arm
        # A_a: (d x d) matrix for each arm a. Stores (X_a^T X_a + lambda_reg * I)
        # b_a: (d x 1) vector for each arm a. Stores (X_a^T y_a)
        self.A = [np.identity(self.n_features) * self.lambda_reg for _ in range(self.n_arms)]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]

    def update(self, chosen_arm_idx, context_x, reward):
        """
        Updates the parameters for the chosen arm.

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

