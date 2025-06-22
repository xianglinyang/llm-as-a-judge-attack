'''Handle the reward calculation for the bandit agent

1. consider whether it is relative or absolute
2. consider whether it is pointwise or pairwise
'''
from src.llm_evaluator import load_judge_model, JudgeType
from abc import abstractmethod
import logging
from enum import Enum
from typing import Union, List, Tuple, Optional

logger = logging.getLogger(__name__)

class RewardType(Enum):
    """Enumeration for reward calculation types"""
    RELATIVE = "relative"
    ABSOLUTE = "absolute"

class RewardCalculatorABC:
    def __init__(self, judge_type: JudgeType, judge_model_backbone: str, reward_type: Union[str, RewardType] = "relative"):
        self.judge_type = judge_type
        self.judge_model_backbone = judge_model_backbone
        self.llm_evaluator = load_judge_model(judge_type, judge_model_backbone)
        self.reward_type = RewardType(reward_type) if isinstance(reward_type, str) else reward_type

    @abstractmethod
    def calculate_reward(self, **kwargs) -> Tuple[float, float, str]:
        """Calculate reward for a single response"""
        pass

    @abstractmethod
    def calculate_batch_reward(self) -> Tuple[List[float], List[float], List[str]]:
        """Calculate rewards for a batch of responses"""
        pass

class PointwiseRewardCalculator(RewardCalculatorABC):
    def __init__(self, judge_type: JudgeType, judge_model_backbone: str, reward_type: Union[str, RewardType] = "relative"):
        super().__init__(judge_type, judge_model_backbone, reward_type)
    
    def calculate_reward(self, 
                         question: str, 
                         response: str, 
                         original_score: float,
                         **kwargs) -> Tuple[float, float, str]:
        """
        Calculate reward for a single response using pointwise evaluation.
        
        Args:
            question (str): The input question
            response (str): The current response to evaluate
            original_score (float): The original score for comparison
            **kwargs: Additional arguments (not used for pointwise)
            
        Returns:
            tuple[float, float, str]: (reward, score, explanation)
                - reward: The calculated reward value
                - score: The absolute score of current response
                - explanation: The explanation for the score
        """
        # Get pointwise score for the current response
        score, explanation = self.llm_evaluator.get_score(question, response)
        
        # Calculate reward based on reward type
        if self.reward_type == RewardType.RELATIVE:
            reward = score - original_score
        elif self.reward_type == RewardType.ABSOLUTE:
            reward = score
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
            
        return reward, score, explanation
    
    def calculate_batch_reward(self, 
                               question_list: List[str], 
                               response_list: List[str], 
                               original_score_list: List[float], 
                               **kwargs) -> Tuple[List[float], List[float], List[str]]:
        """
        Calculate rewards for a batch of responses using pointwise evaluation.
        
        Args:
            question_list (List[str]): List of input questions
            response_list (List[str]): List of current responses to evaluate
            original_score_list (List[float]): List of original scores
            **kwargs: Additional arguments (not used for pointwise)
            
        Returns:
            tuple[List[float], List[float], List[str]]: (reward_list, score_list, explanation_list)
                - reward_list: List of calculated reward values
                - score_list: List of absolute scores for current responses
                - explanation_list: List of explanations for the scores
        """
        # Get pointwise scores for all responses
        score_list, explanation_list = self.llm_evaluator.batch_get_score(question_list, response_list)
        
        # Calculate rewards based on reward type
        if self.reward_type == RewardType.RELATIVE:
            reward_list = [score - original_score for score, original_score in zip(score_list, original_score_list)]
        elif self.reward_type == RewardType.ABSOLUTE:
            reward_list = score_list
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
            
        return reward_list, score_list, explanation_list


class PairwiseRewardCalculator(RewardCalculatorABC):
    def __init__(self, judge_type: JudgeType, judge_model_backbone: str, reward_type: Union[str, RewardType] = "absolute"):
        super().__init__(judge_type, judge_model_backbone, reward_type)
    
    def calculate_reward(self, 
                         question: str, 
                         response: str, 
                         original_score: float, 
                         baseline_response: str, 
                         **kwargs) -> Tuple[float, float, str]:
        """
        Calculate reward for a single response using pairwise comparison.
        
        Args:
            question (str): The input question
            response (str): The current response to evaluate
            original_score (float): The original score for comparison
            baseline_response (str): The baseline response to compare against
            **kwargs: Additional arguments
            
        Returns:
            tuple[float, float, str]: (reward, score, explanation)
                - reward: The calculated reward value
                - score: The normalized pairwise score
                - explanation: The explanation for the score
        """
        # Use pairwise comparison
        raw_score, explanation = self.llm_evaluator.get_score(question, response, baseline_response)
        
        # Calculate reward based on reward type
        if self.reward_type == RewardType.RELATIVE:
            reward = raw_score - original_score
        elif self.reward_type == RewardType.ABSOLUTE:
            reward = raw_score
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
            
        return reward, raw_score, explanation
    
    def calculate_batch_reward(self, 
                               question_list: List[str], 
                               response_list: List[str], 
                               original_score_list: List[float], 
                               baseline_response_list: Optional[List[str]], 
                               **kwargs) -> Tuple[List[float], List[float], List[str]]:
        """
        Calculate rewards for a batch of responses using pairwise comparison.
        
        Args:
            question_list (List[str]): List of input questions
            response_list (List[str]): List of current responses to evaluate
            original_score_list (List[float]): List of original scores
            baseline_response_list (Optional[List[str]]): List of baseline responses to compare against
            **kwargs: Additional arguments
            
        Returns:
            tuple[List[float], List[float], List[str]]: (reward_list, score_list, explanation_list)
                - reward_list: List of calculated reward values
                - score_list: List of normalized pairwise scores
                - explanation_list: List of explanations for the scores
        """
        if baseline_response_list is None:
            raise ValueError("baseline_response_list is required for pairwise evaluation")
            
        # Get pairwise scores for all responses
        raw_scores, explanations = self.llm_evaluator.batch_get_score(
            question_list, response_list, baseline_response_list
        )

        # Calculate rewards based on reward type
        if self.reward_type == RewardType.RELATIVE:
            reward_list = [raw_score - original_score 
                          for raw_score, original_score in zip(raw_scores, original_score_list)]
        elif self.reward_type == RewardType.ABSOLUTE:
            reward_list = raw_scores
        else:
            raise ValueError(f"Invalid reward type: {self.reward_type}")
            
        return reward_list, raw_scores, explanations


def create_reward_calculator(
        judge_type: JudgeType, 
        judge_model_backbone: str, 
        reward_type: Union[str, RewardType] = "relative",
        **kwargs) -> RewardCalculatorABC:
    """
    Factory function to create the appropriate reward calculator based on judge type.
    
    Args:
        judge_type (str): Type of judge ("pointwise", "pairwise", "alpaca_eval", "arena_hard_auto", etc.)
        llm_evaluator (JudgeModel): The judge model instance
        reward_type (Union[str, RewardType]): Type of reward calculation ("relative" or "absolute")
        pairwise_granularity (Union[str, PairwiseGranularity]): Granularity for pairwise comparison
        
    Returns:
        RewardCalculatorABC: The appropriate reward calculator instance
        
    Raises:
        ValueError: If judge_type is not supported
    """
    # Determine if the judge type supports pointwise or pairwise evaluation
    if judge_type in [JudgeType.POINTWISE, JudgeType.MT_BENCH]:
        return PointwiseRewardCalculator(judge_type, judge_model_backbone, reward_type)
    elif judge_type in [JudgeType.PAIRWISE, JudgeType.ALPACA_EVAL, JudgeType.ARENA_HARD_AUTO]:
        return PairwiseRewardCalculator(judge_type, judge_model_backbone, reward_type)
    else:
        # Default to pairwise for unknown judge types
        logger.warning(f"Unknown judge_type '{judge_type}', defaulting to pointwise evaluation")
        return PointwiseRewardCalculator(judge_type, judge_model_backbone, reward_type)


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Test the pointwise reward calculator
    # ------------------------------------------------------------
    judge_type = JudgeType.POINTWISE
    judge_model_backbone = "gpt-4.1"
    reward_type = RewardType.RELATIVE
    reward_calculator = create_reward_calculator(judge_type, judge_model_backbone, reward_type)
    
    
    question = "I'm curious about how renewable energy technologies might transform urban infrastructure in the next decade. What potential innovations do you see emerging?"
    response = "Renewable energy technologies are poised to revolutionize urban landscapes, with advanced solar roadways potentially generating electricity while supporting vehicle traffic. Vertical wind turbines integrated into skyscrapers could become more efficient, potentially producing up to 30% of a building's energy needs. While some experts are skeptical about total urban energy independence, emerging technologies like transparent solar panels and bio-responsive energy systems suggest we're approaching a transformative period in sustainable urban design."
    original_score = -1
    reward, score, explanation = reward_calculator.calculate_reward(question, response, original_score)
    print(f"Reward: {reward}, Score: {score}, Explanation: {explanation}")

    # ------------------------------------------------------------
    # Test the pairwise reward calculator
    # ------------------------------------------------------------
    judge_type = JudgeType.PAIRWISE
    judge_model_backbone = "gemini-2.0-flash"
    reward_type = RewardType.RELATIVE
    reward_calculator = create_reward_calculator(judge_type, judge_model_backbone, reward_type)

    baseline_response = "ChatGPT: Sustainable urban design is a critical component of urban planning, aiming to create cities that are environmentally friendly, socially equitable, and economically viable. This approach considers the long-term impact of urban development on the natural environment, human well-being, and economic prosperity. Key principles include reducing carbon emissions, conserving water resources, promoting biodiversity, and ensuring access to green spaces. By integrating sustainable practices into urban design, cities can achieve a balance between economic growth and environmental sustainability, fostering a more resilient and livable urban environment."
    response = "Renewable energy technologies are poised to revolutionize urban landscapes, with advanced solar roadways potentially generating electricity while supporting vehicle traffic. Vertical wind turbines integrated into skyscrapers could become more efficient, potentially producing up to 30% of a building's energy needs. While some experts are skeptical about total urban energy independence, emerging technologies like transparent solar panels and bio-responsive energy systems suggest we're approaching a transformative period in sustainable urban design."
    original_score = -1
    reward, score, explanation = reward_calculator.calculate_reward(question, response, original_score, baseline_response)
    print(f"Reward: {reward}, Score: {score}, Explanation: {explanation}")



