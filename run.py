from nn import NeuralNetwork
from population import Population
from tasks import DiabetesEvaluator


if __name__ == "__main__":
    GENERATIONS = 1000
    population = Population(evaluator=DiabetesEvaluator())

    for generation in range(GENERATIONS):
        population.evolve()
        print(f"gen #{generation} | max fitness: {population.champions[-1].fitness}")
        test_nn = NeuralNetwork(population.champions[-1])
        if generation % 10 == 0:
            test_nn.visualize()

        if generation % 100 == 0:
            population.evaluator.log_results(test_nn)

        if population.solved_at is not None or generation == GENERATIONS-1:
            population.evaluator.log_results(test_nn)
            test_nn.visualize()
            break
