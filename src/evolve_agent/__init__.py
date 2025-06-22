'''ABC of the evolve agent'''

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
from src.llm_zoo import ModelWrapper
from src.llm_evaluator import JudgeType
from src.evolve_agent.bandit.reward_cal import RewardType, create_reward_calculator

class EvolveAgent(ABC):
    def __init__(self, 
                 llm_agent: ModelWrapper,
                 judge_type: JudgeType,
                 judge_model_backbone: str,
                 reward_type: Union[str, RewardType] = "relative",
                 answer_position: str = "first"):
        """
        Initialize the base evolve agent.
        
        Args:
            llm_agent (ModelWrapper): LLM agent for generating responses
            judge_type (JudgeType): Type of judge for evaluation
            judge_model_backbone (str): Backbone model for the judge
            reward_type (Union[str, RewardType]): Type of reward calculation
            answer_position (str): The position of the answer in the pairwise comparison
        """
        self.evolve_strategy = "ABCStrategy"
        self.llm_agent = llm_agent
        self.judge_type = judge_type
        self.judge_model_backbone = judge_model_backbone
        self.reward_type = reward_type
        self.answer_position = answer_position
        
        # Create reward calculator based on judge type
        self.reward_calculator = create_reward_calculator(
            judge_type=judge_type,
            judge_model_backbone=judge_model_backbone,
            reward_type=reward_type,
            answer_position=answer_position
        )
    
    @abstractmethod
    def explore(self, 
                question: str, 
                init_response: str, 
                original_score: float, 
                original_explanation: str, 
                budget: int = 5, 
                pool_size: int = 2,
                baseline_response: Optional[str] = None,
                **kwargs) -> List[tuple]:
        """
        Explore and evolve a single response.
        
        Args:
            question (str): The input question
            init_response (str): Initial response to evolve
            original_score (float): Original score of the response
            original_explanation (str): Explanation for the original score
            budget (int): Number of exploration steps
            pool_size (int): Maximum size of response pool
            baseline_response (Optional[str]): Baseline response for pairwise evaluation
            **kwargs: Additional arguments for specific implementations
            
        Returns:
            List[tuple]: List of (score, explanation, response, strategy) tuples representing the evolution trajectory
        """
        pass

    async def batch_explore(self, 
                            question_list: List[str], 
                            init_response_list: List[str], 
                            original_score_list: List[float], 
                            original_explanation_list: List[str], 
                            budget: int = 5, 
                            pool_size: int = 2, 
                            baseline_response_list: Optional[List[str]] = None, **kwargs) -> List[List[tuple]]:
        """
        Batch explore the responses
        """
        pass

    # @abstractmethod
    # def online_learning(self, 
    #                    question_list: List[str],
    #                    init_response_list: List[str], 
    #                    original_score_list: List[float], 
    #                    original_explanation_list: List[str],
    #                    budget: int = 5, 
    #                    pool_size: int = 2,
    #                    baseline_response_list: Optional[List[str]] = None,
    #                    **kwargs) -> List[List[tuple]]:
    #     """
    #     Perform online learning across multiple questions.
        
    #     Args:
    #         question_list (List[str]): List of input questions
    #         init_response_list (List[str]): List of initial responses
    #         original_score_list (List[float]): List of original scores
    #         original_explanation_list (List[str]): List of original explanations
    #         budget (int): Number of exploration steps
    #         pool_size (int): Maximum size of response pool
    #         baseline_response_list (Optional[List[str]]): List of baseline responses for pairwise evaluation
    #         **kwargs: Additional arguments for specific implementations
            
    #     Returns:
    #         List[List[tuple]]: List of evolution trajectories for each question
    #     """
    #     pass
    
    def get_reward(self, 
                   question: str, 
                   response: str, 
                   original_score: float, 
                   baseline_response: Optional[str] = None) -> tuple[float, float, str]:
        """
        Get reward for a single response using the configured reward calculator.
        
        Args:
            question (str): The input question
            response (str): The response to evaluate
            original_score (float): Original score for comparison
            baseline_response (Optional[str]): Baseline response for pairwise evaluation
            
        Returns:
            tuple[float, float, str]: (reward, score, explanation)
        """
        if self.judge_type in [JudgeType.POINTWISE, JudgeType.MT_BENCH]:
            return self.reward_calculator.calculate_reward(
                question=question, 
                response=response, 
                original_score=original_score
            )
        elif self.judge_type in [JudgeType.PAIRWISE, JudgeType.ALPACA_EVAL, JudgeType.ARENA_HARD_AUTO]:
            if baseline_response is None:
                raise ValueError(f"baseline_response is required for {self.judge_type} evaluation")
            return self.reward_calculator.calculate_reward(
                question=question, 
                response=response, 
                original_score=original_score,
                baseline_response=baseline_response
            )
        else:
            raise ValueError(f"Unsupported judge type: {self.judge_type}")
    
    async def get_batch_reward(self, 
                        question_list: List[str], 
                        response_list: List[str], 
                        original_score_list: List[float], 
                        baseline_response_list: Optional[List[str]] = None) -> tuple[List[float], List[float], List[str]]:
        """
        Get rewards for a batch of responses.
        
        Args:
            question_list (List[str]): List of input questions
            response_list (List[str]): List of responses to evaluate
            original_score_list (List[float]): List of original scores
            baseline_response_list (Optional[List[str]]): List of baseline responses for pairwise evaluation
            
        Returns:
            tuple[List[float], List[float], List[str]]: (reward_list, score_list, explanation_list)
        """
        if self.judge_type in [JudgeType.POINTWISE, JudgeType.MT_BENCH]:
            return await self.reward_calculator.calculate_batch_reward(
                question_list=question_list,
                response_list=response_list,
                original_score_list=original_score_list
            )
        elif self.judge_type in [JudgeType.PAIRWISE, JudgeType.ALPACA_EVAL, JudgeType.ARENA_HARD_AUTO]:
            if baseline_response_list is None:
                raise ValueError(f"baseline_response_list is required for {self.judge_type} evaluation")
            return await self.reward_calculator.calculate_batch_reward(
                question_list=question_list,
                response_list=response_list,
                original_score_list=original_score_list,
                baseline_response_list=baseline_response_list
            )
        else:
            raise ValueError(f"Unsupported judge type: {self.judge_type}")
    
    def validate_judge_requirements(self, baseline_response: Optional[str] = None, baseline_response_list: Optional[List[str]] = None):
        """
        Validate that required parameters are provided for the judge type.
        
        Args:
            baseline_response (Optional[str]): Baseline response for single evaluation
            baseline_response_list (Optional[List[str]]): Baseline responses for batch evaluation
        """
        if self.judge_type in [JudgeType.PAIRWISE, JudgeType.ALPACA_EVAL, JudgeType.ARENA_HARD_AUTO]:
            if baseline_response is None and baseline_response_list is None:
                raise ValueError(f"baseline_response is required for {self.judge_type} evaluation")
            elif baseline_response_list is not None and None in baseline_response_list:
                raise ValueError(f"All baseline_response_list elements must be non-None for {self.judge_type} evaluation")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the agent configuration.
        
        Returns:
            Dict[str, Any]: Agent configuration information
        """
        return {
            "evolve_strategy": self.evolve_strategy,
            "judge_type": self.judge_type.value,
            "judge_model_backbone": self.judge_model_backbone,
            "reward_type": self.reward_type.value if isinstance(self.reward_type, RewardType) else self.reward_type,
            "reward_calculator_type": type(self.reward_calculator).__name__
        }

    