'''ABC of the evolve agent'''

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

    