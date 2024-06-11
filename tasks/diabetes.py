import numpy as np

from nn import NeuralNetwork
from tasks.evaluator import Evaluator


def init_dataset():
    file_path = "data/diabetes/diabetes1.dt"
    data_input = []
    data_output = []
    with open(file_path, "r") as file:
        for line in file:
            values = line.strip().split(" ")
            try:
                v = [float(value) for value in values[:8]]
                answers = [int(value) for value in values[8:]]
            except ValueError:
                continue
            data_input.append(v)
            data_output.append(answers)

    return data_input, data_output


class DiabetesEvaluator(Evaluator):
    def __init__(self):
        self.tolerance = 0.8
        data_input, data_output = init_dataset()
        self.data_input = data_input
        self.data_output = data_output
        self.train_data = data_input[:200]
        self.test_data = data_input[200:]
        self.train_answers = data_output[:200]
        self.test_answers = data_output[200:]

    def evaluate(self, neural_network: NeuralNetwork) -> float:
        total_fitness = 0
        right_ans = 0
        wrong_ans = 0
        for input_vector, expected_output in zip(self.train_data, self.train_answers):
            predicted_output = neural_network.feed(input_vector)
            error = np.sum(
                (np.abs(np.array(predicted_output) - np.array(expected_output))) ** 2
            ).mean()

            fitness = 1 / (1 + error)
            total_fitness += fitness

            if all(
                np.abs(np.array(predicted_output) - np.array(expected_output)) < 0.5
            ):
                right_ans += 1
            else:
                wrong_ans += 1

        return total_fitness / len(self.train_data)

    def solve(self, neural_network: NeuralNetwork) -> bool:
        right_ans = 0
        wrong_ans = 0
        for input_vector, expected_output in zip(self.train_data, self.train_answers):
            predicted_output = neural_network.feed(input_vector)
            if all(
                np.abs(np.array(predicted_output) - np.array(expected_output)) < 0.5
            ):
                right_ans += 1
            else:
                wrong_ans += 1
        return right_ans / (right_ans + wrong_ans) > self.tolerance

    def log_results(self, neural_network: NeuralNetwork):
        right_ans = 0
        wrong_ans = 0
        for input_vector, expected_output in zip(self.test_data, self.test_answers):
            predicted_output = neural_network.feed(input_vector)

            if all(
                np.abs(np.array(predicted_output) - np.array(expected_output)) < 0.5
            ):
                right_ans += 1
            else:
                wrong_ans += 1

        print(f"right answers: {right_ans}; wrong answers: {wrong_ans}")
