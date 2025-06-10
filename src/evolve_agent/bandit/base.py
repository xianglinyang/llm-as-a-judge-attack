'''Base class for Contextual Bandit Agents.

TODO: Can try Neural Bandit in the future.
'''
from src.evolve_agent import EvolveAgent
from src.llm_zoo import ModelWrapper
from src.text_encoder import TextEncoder
from src.llm_evaluator import JudgeModel

# TODO: Add more methods for the base class.
class ContextualBanditAgent(EvolveAgent):
    def __init__(self, n_features: int, llm_agent: ModelWrapper, embedding_model: TextEncoder, llm_evaluator: JudgeModel):
        super().__init__()
        self.n_features = n_features
        self.llm_agent = llm_agent
        self.embedding_model = embedding_model
        self.llm_evaluator = llm_evaluator
    
    def get_context_x(self, question: str, response: str):
        pass
    
    def get_reward(self, question: str, response: str, original_score: float):
        pass
    
    def get_policy_distribution(self, context_x):
        pass
    
    def choose_arm(self, context_x): 
        pass

    def explore(self, question, init_response, pool_size: int, Budget: int):
        pass
    
    def explore_with_random_arm(self, question, init_response, pool_size: int, Budget: int):
        pass
    