'''ABC of the evolve agent'''

from abc import ABC, abstractmethod
class EvolveAgent(ABC):
    def __init__(self):
        self.evolve_strategy = "ABCStrategy"
    
    @abstractmethod
    def explore(self):
        pass

    @abstractmethod
    def online_learning(self):
        pass

    