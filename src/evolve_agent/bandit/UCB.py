import numpy as np
import random
import pickle
import json
import os
from tqdm import tqdm

from src.evolve_agent import EvolveAgent
from src.evolve_agent.bandit.config import strategy_list, STRATEGY_PROMPT, TEST_STRATEGY_PROMPT
from src.llm_evaluator import JudgeModel
from src.text_encoder import TextEncoder, MiniLMTextEncoder
from src.llm_zoo import ModelWrapper, load_model
from src.utils import str2json

class ContextualLinUCBAgent(EvolveAgent):
    def __init__(self, n_features: int, llm_agent: ModelWrapper, embedding_model: TextEncoder, llm_evaluator: JudgeModel, init_model_name: str, alpha: float = 1.0, lambda_reg: float = 1.0):
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
        self.init_model_name = init_model_name

        self.init_policy()
    
    def save_policy(self):
        with open(f"policy_{self.llm_evaluator.model_name}_{self.llm_agent.model_name}.pkl", "wb") as f:
            pickle.dump(self.A, f)
            pickle.dump(self.b, f)
    
    def load_policy(self):
        with open(f"policy_{self.llm_evaluator.model_name}_{self.llm_agent.model_name}.pkl", "rb") as f:
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
        for question, response, _ in datasets:
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
            # prompt = TEST_STRATEGY_PROMPT.format(question=question, 
            #                                     response=response, 
            #                                     N=1, 
            #                                     strategy=strategy_list[chosen_arm],
            #                                     )
            
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
categories = [
    "Computer Science & Programming",
    "Mathematics & Statistics",
    "Science & Engineering",
    "Business & Finance",
    "Writing & Communication",
    "Social & Daily Life",
    "Others"
]

def prepare_dataset(train_num, test_num):

    # Data stream for training
    path = "/mnt/hdd1/ljiahao/xianglin/llm-as-a-judge-attack/data/UltraFeedback/UltraFeedback_category.json"

    import json
    with open(path, 'r') as f:
        data = json.load(f)

    # prepare dataset
    dataset = []
    for item in data:
        try:
            question = item["question"]
            category = item["categorization"]["question category"]
            if category in categories:
                dataset.append((question, category))
        except:
            continue
    
    # split dataset into train and test
    idxs = np.random.choice(len(dataset), size=train_num+test_num, replace=False)
    train_dataset = [dataset[i] for i in idxs[:train_num]]
    test_dataset = [dataset[i] for i in idxs[train_num:]]
    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_num", type=int, default=2)
    parser.add_argument("--test_num", type=int, default=1)
    parser.add_argument("--train_T", type=int, default=2)
    parser.add_argument("--eval_T", type=int, default=10)
    parser.add_argument("--judge_model_name", type=str, default="gemini-1.5-flash")
    parser.add_argument("--llm_agent_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--init_answer_model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--n_features", type=int, default=384)
    parser.add_argument("--save_path", type=str, default="output/")
    parser.add_argument("--test_mode", type=str, default="policy", choices=["random", "policy"])

    args = parser.parse_args()

    train_num = args.train_num
    test_num = args.test_num

    n_features = args.n_features
    llm_agent = load_model(args.llm_agent_name)
    embedding_model = MiniLMTextEncoder()
    judge_model_name = args.judge_model_name
    llm_evaluator = JudgeModel(judge_model_name)
    alpha = args.alpha
    lambda_reg = args.lambda_reg

    train_T = args.train_T
    eval_T = args.eval_T

    train_dataset, test_dataset = prepare_dataset(train_num, test_num)

    init_answer_model_name = args.init_answer_model_name
    init_answer_model = load_model(init_answer_model_name)

    new_train_dataset = []
    for question, category in train_dataset:
        answer = init_answer_model.invoke(question)
        new_train_dataset.append((question, answer, category))
    
    new_test_dataset = []
    for question, category in test_dataset:
        answer = init_answer_model.invoke(question)
        new_test_dataset.append((question, answer, category))

    agent = ContextualLinUCBAgent(n_features, llm_agent, embedding_model, llm_evaluator, init_answer_model_name, alpha, lambda_reg)
    
    if args.test_mode == "policy":
        agent.train_policy(new_train_dataset, train_T)

    # Test the policy
    test_results = []
    for question, response, category in new_test_dataset:
        if args.test_mode == "policy":
            new_response = agent.test_policy(question, response, eval_T)
        elif args.test_mode == "random":
            new_response = agent.test_random_policy(question, response, eval_T)
        else:
            raise ValueError(f"Invalid test mode: {args.test_mode}")

        # Evaluate the policy
        original_score, original_explanation = llm_evaluator.pointwise_score(question, response)
        new_score, new_explanation = llm_evaluator.pointwise_score(question, new_response)
        print(f"Original score: {original_score}, explanation: {original_explanation}")
        print(f"New score: {new_score}, explanation: {new_explanation}")
        test_results.append((question, response, category, original_score, original_explanation, new_score, new_explanation))
        print("--------------------------------")
    
    # save the test results
    with open(os.path.join(args.save_path, f"{args.test_mode}_{judge_model_name}_{llm_agent.model_name}_{init_answer_model_name}_test_results.json"), "w") as f:
        json.dump(test_results, f)

    analysis = {}
    # Analyze the test result for each category
    for category in categories:
        category_results = [result for result in test_results if result[2] == category]
        print(f"Category: {category}")
        print(f"Number of results: {len(category_results)}")
        up_num = len([result for result in category_results if result[4] < result[6]])
        down_num = len([result for result in category_results if result[4] > result[6]])
        tie_num = len([result for result in category_results if result[4] == result[6]])
        print(f"Number of up results: {up_num}")
        print(f"Number of down results: {down_num}")
        print(f"Number of tie results: {tie_num}")
        # print(f"Average score: {np.mean([result[4] for result in category_results])}")
        # print(f"Average new score: {np.mean([result[6] for result in category_results])}")
        # print(f"Average improvement: {np.mean([result[6] - result[4] for result in category_results if result[4] < result[6]])}")
        print("--------------------------------")
        analysis[category] = {
            "up_num": up_num,
            "down_num": down_num,
            "tie_num": tie_num,
            # "average_score": np.mean([result[4] for result in category_results]),
            # "average_new_score": np.mean([result[6] for result in category_results]),
        }
    
    # save the analysis
    with open(os.path.join(args.save_path, f"{args.test_mode}_{judge_model_name}_{llm_agent.model_name}_{init_answer_model_name}_analysis.json"), "w") as f:
        json.dump(analysis, f)
        