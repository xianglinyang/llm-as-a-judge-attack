'''Base class for Contextual Bandit Agents.

TODO: Can try Neural Bandit in the future.
'''
import numpy as np
import random
import heapq
import logging
import argparse
from tqdm import tqdm
import os
import json
from abc import abstractmethod
from sklearn.utils import shuffle
from typing import Callable

from src.evolve_agent import EvolveAgent
from src.llm_zoo import ModelWrapper
from src.text_encoder import TextEncoder
from src.llm_evaluator import JudgeModel
from src.text_encoder import TextEncoder
from src.evolve_agent.bias_strategies import Bias_types, BiasModification

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

class ContextualBanditAgent(EvolveAgent):
    def __init__(self, n_features: int, llm_agent: ModelWrapper, embedding_model: TextEncoder, llm_evaluator: JudgeModel, reward_type: str = "relative"):
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
        self.bias_modificatior = BiasModification(llm_agent)
    
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
    
    def get_reward(self, question: str, response: str, original_score: float):
        s, e = self.llm_evaluator.pointwise_score(question, response)
        if self.reward_type == "relative":
            reward = s-original_score
        elif self.reward_type == "absolute":
            reward = s
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
        return reward, s, e
    
    def explore(self, question, init_response, pool_size: int, Budget: int, cold_start: bool):
        if Budget<self.n_arms:
            raise ValueError(f"Budget must be greater than the number of arms: {self.n_arms}")
        # init the response pool
        pool = []
        s, e = self.llm_evaluator.pointwise_score(question, init_response)
        pool.append([s, (s, e, init_response)])

        curr_step = 0
        if cold_start:
            for arm_idx in range(self.n_arms):
                context_x = self.get_context_x(question, init_response)
                context_x = context_x.reshape(-1, 1)

                new_response = self.bias_modificatior.principle_guided_mutation(init_response, Bias_types[arm_idx])

                reward, new_score, new_explanation = self.get_reward(question, new_response, s)
                self.update(arm_idx, context_x, reward)
                
                pool.append([s, (s, e, init_response), (new_score, new_explanation, new_response)])
                if len(pool) > pool_size:
                    heapq.heappop(pool)

                # record the results
                logger.info(f"Iteration {arm_idx}:")
                logger.info(f"Original score: {s}, explanation: {e}")
                logger.info(f"New score: {new_score}, explanation: {new_explanation}")
                logger.info(f"Chosen arm: {self.strategy_list[arm_idx]}")
                logger.info(f"New response: {new_response}\n")
                curr_step += 1
        
        for t in range(curr_step, Budget):
            # 1. Sample a context from the dataset
            idx = random.choice(range(len(pool)))
            
            curr_path = pool[idx].copy()
            curr_s, curr_e, curr_r = curr_path[-1]

            context_x = self.get_context_x(question, curr_r)
            context_x = context_x.reshape(-1, 1)

            # 2. Choose an arm
            chosen_arm = self.choose_arm(context_x)

            # 3. get new response
            new_response = self.bias_modificatior.principle_guided_mutation(curr_r, Bias_types[chosen_arm])

            # 4. Get the reward
            reward, new_score, new_explanation = self.get_reward(question, new_response, curr_s)

            # 5. Update the policy
            self.update(chosen_arm, context_x, reward)

            # 5.1 update the pool with heapq
            curr_path.append((new_score, new_explanation, new_response))
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

        best_path = find_shortest_of_max_simple(pool)
        # final_score, final_explanation, final_response = best_path[-1]
        return best_path
    
    def explore_with_random_arm(self, question, init_response, pool_size: int, Budget: int):
        # init the response pool
        pool = []
        s, e = self.llm_evaluator.pointwise_score(question, init_response)
        pool.append([s, (s, e, init_response)])
        
        for t in range(Budget):
            # 1. Sample a context from the dataset
            idx = random.choice(range(len(pool)))
            curr_path = pool[idx].copy()
            curr_s, curr_e, curr_r = curr_path[-1]

            # 2. Choose an arm
            chosen_arm = random.choice(range(self.n_arms))

            # 3. get new response
            new_response = self.bias_modificatior.principle_guided_mutation(curr_r, Bias_types[chosen_arm])

            # 4. Get the reward
            reward, new_score, new_explanation = self.get_reward(question, new_response, curr_s)

            # 5.1 update the pool with heapq
            curr_path.append((new_score, new_explanation, new_response))
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
    
    def online_learning(self, question_list, init_response_list, pool_size: int, Budget: int, shuffle_data: bool = True, init_policy: bool = True):
        # shuffle the question and init_response (Optional)
        if shuffle_data:
            logger.info(f"Shuffling the question and init_response")
            question_list, init_response_list = shuffle(question_list, init_response_list)
        
        if init_policy:
            logger.info(f"Initializing the policy model")
            self.init_policy_model()
        
        explore_trajectories = []
        # online learning
        logger.info(f"Online learning started")
        for t, (question, init_response) in tqdm(enumerate(zip(question_list, init_response_list))):
            explore_trajectory = self.explore(question, init_response, pool_size, Budget, cold_start=True if t==0 else False)
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

