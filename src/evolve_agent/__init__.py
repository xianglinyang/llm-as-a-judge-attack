'''ABC of the evolve agent'''

# from src.evolve_agent.LinUCB import ContextualLinUCBAgent, ContextualNeuralUCBAgent
# from src.evolve_agent.bandit.bandit_agent import TS_MAB_EvolveAgent


# # load all agent here
# ALL_AGENTS = {
#     "LinUCB": ContextualLinUCBAgent,
#     "NeuralUCB": ContextualNeuralUCBAgent,
# }



from abc import ABC, abstractmethod
class EvolveAgent(ABC):
    def __init__(self):
        self.evolve_strategy = "ABCStrategy"
    
    @abstractmethod
    def train_policy(self):
        pass
    
    @abstractmethod
    def test_policy(self):
        pass

    