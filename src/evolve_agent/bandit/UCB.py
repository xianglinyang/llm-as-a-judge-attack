import numpy as np
import random
import pickle

from src.evolve_agent import EvolveAgent
from src.evolve_agent.bandit.config import strategy_list, STRATEGY_PROMPT, TEST_STRATEGY_PROMPT
from src.llm_evaluator import JudgeModel
from src.text_encoder import TextEncoder, MiniLMTextEncoder
from src.llm_zoo import ModelWrapper, load_model
from src.utils import str2json

class ContextualLinUCBAgent(EvolveAgent):
    def __init__(self, n_features: int, llm_agent: ModelWrapper, embedding_model: TextEncoder, llm_evaluator: JudgeModel, alpha: float = 1.0, lambda_reg: float = 1.0):
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
        super().__init__()

        self.n_arms = len(strategy_list) # number of arms
        self.strategy_list = strategy_list
        self.n_features = n_features
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.llm_evaluator = llm_evaluator
        self.embedding_model = embedding_model
        self.llm_agent = llm_agent

        self.init_policy()
    
    def save_policy(self):
        with open("policy.pkl", "wb") as f:
            pickle.dump(self.A, f)
            pickle.dump(self.b, f)
    
    def load_policy(self):
        with open("policy.pkl", "rb") as f:
            self.A = pickle.load(f)
            self.b = pickle.load(f)
    
    def init_policy(self):
        # Initialize parameters for each arm
        # A_a: (d x d) matrix for each arm a. Stores (X_a^T X_a + lambda_reg * I)
        # b_a: (d x 1) vector for each arm a. Stores (X_a^T y_a)
        self.A = [np.identity(self.n_features) * self.lambda_reg for _ in range(self.n_arms)]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]

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
    
    def get_context_x(self, question: str, response: str):
        text = "Question: " + question + "\n" + "Response: " + response
        embedding = self.embedding_model.encode(text)
        return embedding
    
    def get_reward(self, question: str, response: str, original_score: float):
        s, e = self.llm_evaluator.pointwise_score(question, response)
        reward = s-original_score
        return reward, s, e
    
    def get_policy_distribution(self, context_x):
        policy_distribution = []
        for arm_idx in range(self.n_arms):
            A_a_inv = np.linalg.inv(self.A[arm_idx])
            theta_hat_a = A_a_inv @ self.b[arm_idx] # (d x d) @ (d x 1) = (d x 1)
            expected_reward = context_x.T @ theta_hat_a # (1 x d) @ (d x 1) = (1 x 1)
            value = expected_reward
            policy_distribution.append(value)
        return policy_distribution
    
    def show_history(self):
        for policy in self.history_policy_distribution:
            print(policy)

    def train_policy(self, datasets, T: int):
        context_list = []
        history_policy_distribution = []
        for question, response in datasets:
            print(f"Question: {question}")
            print(f"Response: {response}")
            s, e = self.llm_evaluator.pointwise_score(question, response)
            print(f"Original score: {s}, explanation: {e}")
            context_list.append([(question, response, s, e)])
        
        for t in range(T):
            # 1. Sample a context from the dataset
            idx = random.choice(range(len(context_list)))
            context = context_list[idx][-1]
            question, response, original_score, explanation = context

            context_x = self.get_context_x(question, response)
            context_x = context_x.reshape(-1, 1)

            # 2. Choose an arm
            chosen_arm = self.choose_arm(context_x)

            # 3. get new response
            prompt = STRATEGY_PROMPT.format(question=question, 
                                            response=response, 
                                            N=1, 
                                            strategy=strategy_list[chosen_arm],
                                            feedback=explanation,
                                            score=original_score,
                                            )
            
            new_response = self.llm_agent.invoke(prompt)
            new_response = str2json(new_response)[0]

            # 4. Get the reward
            reward, new_score, new_explanation = self.get_reward(question, new_response, original_score)

            # 5. Update the policy
            self.update(chosen_arm, context_x, reward)

            # 5.1 record the value of each arm
            curr_policy = self.get_policy_distribution(context_x)
            history_policy_distribution.append(curr_policy)

            context_list[idx].append((question, new_response, new_score, new_explanation))
            
            # 6. log for evaluation
            print(f"Iteration {t}:")
            print(f"Original score: {original_score}, explanation: {explanation}")
            print(f"New score: {new_score}, explanation: {new_explanation}")
            print(f"Chosen arm: {strategy_list[chosen_arm]}")
            print(f"New response: {new_response}\n")
        
        # save the policy
        self.save_policy()
        
        return context_list
    
    def test_policy(self, question, response, T):
        curr_q, curr_r = question, response
        for t in range(T):
            context_x = self.get_context_x(curr_q, curr_r)
            context_x = context_x.reshape(-1, 1)
            print("Context_x shape: ", context_x.shape)

            chosen_arm = self.choose_arm(context_x)
            prompt = TEST_STRATEGY_PROMPT.format(question=curr_q, 
                                            response=curr_r, 
                                            N=1, 
                                            strategy=strategy_list[chosen_arm],
                                            )
            new_response = self.llm_agent.invoke(prompt)
            new_response = str2json(new_response)[0]
            curr_q, curr_r = question, new_response
        return curr_r
    
    def test_random_policy(self, question, response, T):
        curr_q, curr_r = question, response
        for t in range(T):
            context_x = self.get_context_x(curr_q, curr_r)
            context_x = context_x.reshape(-1, 1)
            print("Context_x shape: ", context_x.shape)

            # chosen_arm = self.choose_arm(context_x)
            chosen_arm = random.choice(range(self.n_arms))
            prompt = TEST_STRATEGY_PROMPT.format(question=curr_q, 
                                            response=curr_r, 
                                            N=1, 
                                            strategy=strategy_list[chosen_arm],
                                            )
            new_response = self.llm_agent.invoke(prompt)
            new_response = str2json(new_response)[0]
            curr_q, curr_r = question, new_response
        return curr_r



# class ContextualNeuralUCBAgent(EvolveAgent):
#     def __init__(self, n_features: int, llm_evaluator: JudgeModel, alpha: float = 1.0, lambda_reg: float = 1.0):
#         """
#         Initializes the NeuralLinUCB agent.
#         """
#         super().__init__()
#         self.n_arms = len(strategy_list) # number of arms



if __name__ == "__main__":
    # Data stream for training
    path = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data/data/human_written/moss_oasst_lima_Llama_Factory.json"

    import json

    with open(path, 'r') as f:
        data = json.load(f)

    dataset = [
        (item["instruction"], item["output"])
        for item in data[:100]
    ]
    n_features = 384
    llm_agent = load_model("gpt-4o-mini")
    embedding_model = MiniLMTextEncoder()
    llm_evaluator = JudgeModel("gemini-1.5-flash")
    alpha = 1.0
    lambda_reg = 1.0

    agent = ContextualLinUCBAgent(n_features, llm_agent, embedding_model, llm_evaluator, alpha, lambda_reg)
    # agent.train_policy(dataset, 100)

    # Test the policy

    for item in data[101:121]:
        question = item["instruction"]
        response = item["output"]
        new_response = agent.test_random_policy(question, response, 10)
        print(new_response)

        # Evaluate the policy
        original_score, original_explanation = llm_evaluator.pointwise_score(question, response)
        new_score, new_explanation = llm_evaluator.pointwise_score(question, new_response)
        print(f"Original score: {original_score}, explanation: {original_explanation}")
        print(f"New score: {new_score}, explanation: {new_explanation}")
        print("--------------------------------")
        