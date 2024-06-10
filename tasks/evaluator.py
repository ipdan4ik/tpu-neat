from nn import NeuralNetwork
from abc import ABC, abstractmethod


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, neural_network: NeuralNetwork) -> float:
        pass

    @abstractmethod
    def solve(self, neural_network: NeuralNetwork) -> float:
        pass

    @abstractmethod
    def log_results(self, neural_network: NeuralNetwork):
        pass
