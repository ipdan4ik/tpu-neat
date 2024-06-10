import numpy as np

from nn import NeuralNetwork


class XOREvaluator:
    def __init__(
        self, tolerance: float = 0.1
    ):
        self.tolerance = tolerance
        self.epsilon = 0.1
        self.inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.expected_outputs = [[0], [1], [1], [0]]

    def evaluate(self, neural_network: NeuralNetwork) -> float:
        total_fitness = 0
        for input_vector, expected_output in zip(self.inputs, self.expected_outputs):
            predicted_output = neural_network.feed(input_vector)
            error = np.sum(np.abs(np.array(predicted_output) - np.array(expected_output)))
            error = error**2
            fitness = 1 / (1 + error)
            total_fitness += fitness
        return total_fitness / len(self.inputs)

    def solve(self, neural_network: NeuralNetwork) -> bool:
        for input_vector, expected_output in zip(self.inputs, self.expected_outputs):
            predicted_output = neural_network.feed(input_vector)
            if not all(np.abs(np.array(predicted_output) - np.array(expected_output)) < 0.5):
                return False
        return True

    def log_results(self, neural_network: NeuralNetwork):
        for input_vector, expected_output in zip(self.inputs, self.expected_outputs):
            predicted_output = neural_network.feed(input_vector)
            print(f"input: {input_vector}; output: {predicted_output}")
