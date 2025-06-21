'''Handle the reward calculation for the bandit agent

1. consider whether it is relative or absolute
2. consider whether it is pointwise or pairwise
'''
from src.llm_evaluator import JudgeModel, load_judge_model
from abc import abstractmethod
import logging

logger = logging.getLogger(__name__)

class RewardCalculatorABC:
    def __init__(self, llm_evaluator: JudgeModel, reward_type: str = "relative"):
        self.llm_evaluator = llm_evaluator
        self.reward_type = reward_type

    @abstractmethod
    def calculate_reward():
        pass

    @abstractmethod
    def calculate_batch_reward():
        pass

class PointwiseRewardCalculator(RewardCalculatorABC):
    def __init__(self, llm_evaluator: JudgeModel, reward_type: str = "relative"):
        super().__init__(llm_evaluator, reward_type)
    
    def calculate_reward(self, question: str, response: str, original_score: float):
        """
        Calculate reward for a single response.
        
        Args:
            question (str): The input question
            response (str): The current response to evaluate
            original_score (float): The original score
            original_response (str, optional): The original response to compare against
            
        Returns:
            tuple[float, float, str]: (reward, score, explanation)
                - reward: The calculated reward value
                - score: The score of current response
                - explanation: The explanation for the score
        """
        # Get pointwise score for the current response
        s, e = self.llm_evaluator.pointwise_score(question, response)
        
        # Calculate reward based on reward type
        if self.reward_type == "relative":
            reward = s - original_score
        elif self.reward_type == "absolute":
            reward = s
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
            
        return reward, s, e
    
    def calculate_batch_reward(self, question_list: list[str], response_list: list[str], original_score_list: list[float]) -> tuple[list[float], list[float], list[str]]:
        """
        Calculate rewards for a batch of responses.
        
        Args:
            question_list (list[str]): List of input questions
            response_list (list[str]): List of current responses to evaluate
            original_score_list (list[float]): List of original scores
            
        Returns:
            tuple[list[float], list[float], list[str]]: (reward_list, score_list, explanation_list)
                - reward_list: List of calculated reward values
                - score_list: List of scores for current responses
                - explanation_list: List of explanations for the scores
        """
        # Get pointwise scores for all responses
        s_list, e_list = self.llm_evaluator.batch_pointwise_score(question_list, response_list)
        
        # Calculate rewards based on reward type
        if self.reward_type == "relative":
            reward_list = [s - original_score for s, original_score in zip(s_list, original_score_list)]
        elif self.reward_type == "absolute":
            reward_list = s_list
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
            
        return reward_list, s_list, e_list


class PairwiseRewardCalculator(RewardCalculatorABC):
    def __init__(self, llm_evaluator: JudgeModel, reward_type: str = "absolute"):
        super().__init__(llm_evaluator, reward_type)
    
    def calculate_reward(self, question: str, response: str, original_score: float, baseline_response: str):
        # Use pairwise comparison
        outcome, explanation = self.llm_evaluator.pairwise_score(question, response, baseline_response)
        if self.reward_type == "relative":
            reward = outcome - original_score
        elif self.reward_type == "absolute":
            reward = outcome
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
        return reward, outcome, explanation
    
    def calculate_batch_reward(self, question_list: list[str], response_list: list[str], original_score_list: list[float], baseline_response_list: list[str] = None) -> tuple[list[float], list[float], list[str]]:
        """
        Calculate rewards for a batch of responses using pairwise comparison.
        
        Args:
            question_list (list[str]): List of input questions
            response_list (list[str]): List of current responses to evaluate
            original_score_list (list[float]): List of original scores
            baseline_response_list (list[str], optional): List of baseline responses to compare against
            
        Returns:
            tuple[list[float], list[float], list[str]]: (reward_list, score_list, explanation_list)
                - reward_list: List of calculated reward values (-2.0 to +2.0)
                - score_list: List of scores for current responses
                - explanation_list: List of explanations for the scores
        """
        outcomes, explanations = self.llm_evaluator.batch_pairwise_score(question_list, response_list, baseline_response_list)
        if self.reward_type == "relative":
            reward_list = [outcome - original_score for outcome, original_score in zip(outcomes, original_score_list)]
        elif self.reward_type == "absolute":
            reward_list = outcomes
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
        return reward_list, outcomes, explanations
    


if __name__ == "__main__":
    llm_evaluator = load_judge_model("general", "gemini-2.0-flash")

    question = "I'm curious about how renewable energy technologies might transform urban infrastructure in the next decade. What potential innovations do you see emerging?"
    response = "Renewable energy technologies are poised to revolutionize urban landscapes, with advanced solar roadways potentially generating electricity while supporting vehicle traffic. Vertical wind turbines integrated into skyscrapers could become more efficient, potentially producing up to 30% of a building's energy needs. While some experts are skeptical about total urban energy independence, emerging technologies like transparent solar panels and bio-responsive energy systems suggest we're approaching a transformative period in sustainable urban design."
    baseline_response = "ChatGPT: Sustainable urban design is a critical component of urban planning, aiming to create cities that are environmentally friendly, socially equitable, and economically viable. This approach considers the long-term impact of urban development on the natural environment, human well-being, and economic prosperity. Key principles include reducing carbon emissions, conserving water resources, promoting biodiversity, and ensuring access to green spaces. By integrating sustainable practices into urban design, cities can achieve a balance between economic growth and environmental sustainability, fostering a more resilient and livable urban environment."

    original_score = -1

    reward_calculator = PairwiseRewardCalculator(llm_evaluator, reward_type="absolute")
    reward, score, explanation = reward_calculator.calculate_reward(question, response, original_score, baseline_response)
    print(f"Reward: {reward}, Score: {score}, Explanation: {explanation}")

    reward_calculator = PointwiseRewardCalculator(llm_evaluator, reward_type="relative")
    reward, score, explanation = reward_calculator.calculate_reward(question, response, original_score)
    print(f"Reward: {reward}, Score: {score}, Explanation: {explanation}")



