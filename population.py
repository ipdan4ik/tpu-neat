import random
from copy import deepcopy

import numpy as np

from configuration import config
from genome import Genome
from main import ConnectionId, innovation_tracker
from nn import NeuralNetwork
from species import Species
from tasks import XOREvaluator
from tasks.evaluator import Evaluator


def mate(genome0: Genome, genome1: Genome) -> Genome:
    child = Genome()

    max_genome = max(genome0, genome1, key=lambda x: len(x.nodes))
    min_genome = min(genome0, genome1, key=lambda x: len(x.nodes))

    for node0, node1 in zip(max_genome.nodes, min_genome.nodes):
        node = random.choice((node0, node1))
        child.nodes.append(deepcopy(node))

    if max_genome.fitness >= min_genome.fitness:
        for node in max_genome.nodes[len(min_genome.nodes):]:
            child.nodes.append(deepcopy(node))

    intersection = genome0.connections.keys() & genome1.connections.keys()

    more_fit_genome = max(genome0, genome1, key=lambda x: x.fitness)
    for connection_id in intersection:
        connection = connection_crossover(connection_id, genome0, genome1)
        child.connections[connection_id] = connection

    if max_genome.fitness == min_genome.fitness:
        connection_union = genome0.connections | genome1.connections
    else:
        connection_union = more_fit_genome.connections

    for connection_id, connection in connection_union.items():
        if connection_id not in intersection:
            child.connections[connection_id] = deepcopy(connection)

    return child


def connection_crossover(connection_id: ConnectionId, genome0: Genome, genome1: Genome):
    connection = deepcopy(genome0.connections[connection_id])
    connection.enabled = (
        genome0.connections[connection_id].enabled
        and genome0.connections[connection_id].enabled
    ) or random.random() > config.prob_reenable_connection

    if config.random_crossover_weights:
        connection.weight = random.choice(
            (
                genome0.connections[connection_id].weight,
                genome1.connections[connection_id].weight,
            )
        )
    else:
        connection.weight = (
            genome0.connections[connection_id].weight
            + genome1.connections[connection_id].weight
        ) / 2
    return connection


def genome_distance(genome0: Genome, genome1: Genome):
    genome0_innovations = {c.innovation_number: c for c in genome0.connections.values()}
    genome1_innovations = {c.innovation_number: c for c in genome1.connections.values()}
    all_innovations = genome0_innovations | genome1_innovations

    min_innovation = min(
        max(genome0_innovations.keys()), max(genome1_innovations.keys())
    )

    excess = 0
    disjoint = 0
    avg_weight_diff = 0.0
    matches = 0

    for i in all_innovations.keys():
        if i in genome0_innovations and i in genome1_innovations:
            avg_weight_diff += np.abs(
                genome0_innovations[i].weight - genome1_innovations[i].weight
            )
            matches += 1
        else:
            if i <= min_innovation:
                disjoint += 1
            else:
                excess += 1

    avg_weight_diff = (avg_weight_diff / matches) if matches > 0 else avg_weight_diff

    return (
        config.distance_excess * excess
        + config.distance_disjoint * disjoint
        + config.distance_weight * avg_weight_diff
    )


def tournament_selection(genomes: list[Genome], k=3) -> Genome:
    selected = random.sample(genomes, k)
    max_fit: tuple[int, Genome] = (0, selected[0])
    for i in selected:
        if i.fitness > max_fit[0]:
            max_fit = (i.fitness, i)
    return max_fit[1]


def create_initial_genome() -> Genome:
    individual = Genome()

    input_nodes = []
    for _ in range(config.input_nodes):
        node = individual.add_node(0)
        input_nodes.append(node)

    if config.add_bias_node:
        individual.add_node(0)

    out_nodes = []
    for _ in range(config.output_nodes):
        node = individual.add_node(config.max_depth + 1)
        out_nodes.append(node)

    for input_node in input_nodes:
        if random.random() < config.initial_connection_prob:
            out_node = random.choice(out_nodes)
            weight = random.uniform(*config.weight_range)
            individual.add_connection(input_node, out_node, weight)
    return individual


class Population:
    def __init__(self):
        self.genomes: list[Genome] = [
            create_initial_genome() for _ in range(config.population_size)
        ]
        self.species: list[Species] = []
        self.current_compatibility_threshold: int = config.compatibility_threshold
        self.champions: list[Genome] = []
        self.evaluator: Evaluator = XOREvaluator()
        self.solved_at: int | None = None
        self.generation_index: int = 0

    def init_species(self):
        for specie in self.species:
            specie.evolve_step()

        for genome in self.genomes:
            for specie in self.species:
                if (
                    genome_distance(genome, specie.representative)
                    < self.current_compatibility_threshold
                ):
                    specie.add_genome(genome)
                    break
            else:
                new_specie = Species(representative=deepcopy(genome))
                new_specie.add_genome(genome)
                self.species.append(new_specie)

        self.species = list(filter(lambda s: len(s.genomes) > 0, self.species))

        if len(self.species) < config.target_species:
            self.current_compatibility_threshold -= config.compatibility_threshold_delta
        elif len(self.species) > config.target_species:
            self.current_compatibility_threshold += config.compatibility_threshold_delta
        if self.current_compatibility_threshold < config.min_compatibility_threshold:
            self.current_compatibility_threshold = config.min_compatibility_threshold

    def evaluate_all_fitness(self):
        for genome in self.genomes:
            genome.fitness = self.evaluate_genome(genome)
        for specie in self.species:
            specie.full_recalculate()

    def evaluate_genome(self, genome: Genome) -> float:
        nn = NeuralNetwork(genome)
        evaluation = self.evaluator.evaluate(nn)
        return evaluation

    def check_for_stagnation(self):
        for specie in self.species:
            specie.recalculate_max_fitness()
            if specie.max_fitness <= specie.prev_max_fitness:
                specie.no_improvement_age += 1
            else:
                specie.no_improvement_age = 0
            specie.has_best = self.champions[-1] in specie.genomes

        species_prev = len(self.species)
        self.species = list(
            filter(
                lambda s: s.no_improvement_age < config.stagnation_age or s.has_best,
                self.species,
            )
        )
        if len(self.species) != species_prev:
            killed_species = species_prev - len(self.species)
            print(f"killed_species: {killed_species}")

    def find_champion(self):
        self.champions.append(max(self.genomes, key=lambda genome: genome.fitness))

    def look_for_solution(self):
        nn = NeuralNetwork(self.champions[-1])
        if self.evaluator.solve(nn):
            self.solved_at = self.generation_index

    def reproduce_offspring(self):
        total_average = sum(specie.avg_adjusted_fitness for specie in self.species)
        for specie in self.species:
            specie.offspring_number = int(
                round(len(self.genomes) * specie.avg_adjusted_fitness / total_average)
            )
        self.species = list(filter(lambda s: s.offspring_number > 0, self.species))
        if config.reset_innovations:
            innovation_tracker.reset_innovations()

        new_genomes_global = []
        for specie in self.species:
            specie.genomes.sort(key=lambda ind: ind.fitness, reverse=True)
            keep = max(1, int(round(len(specie.genomes) * config.genome_survival_rate)))
            pool = specie.genomes[:keep]
            if config.elitism_enabled and len(specie.genomes) >= 1:
                specie.genomes = specie.genomes[:1]
                new_genomes_global += specie.genomes
            else:
                specie.genomes = []

            while len(specie.genomes) < specie.offspring_number:
                new_genomes = []
                if len(pool) == 1:
                    child = deepcopy(pool[0])
                    child.mutate()
                    new_genomes.append(child)
                else:
                    parent1 = deepcopy(
                        tournament_selection(
                            pool, min(len(pool), config.max_tournament_champions)
                        )
                    )
                    parent2 = deepcopy(
                        tournament_selection(
                            pool, min(len(pool), config.max_tournament_champions)
                        )
                    )
                    child = mate(parent1, parent2)
                    child.mutate()
                    new_genomes.append(child)
                specie.genomes += new_genomes
                new_genomes_global += new_genomes
        self.genomes = new_genomes_global

    def evolve(self):
        self.init_species()
        self.evaluate_all_fitness()
        self.find_champion()
        self.look_for_solution()
        self.check_for_stagnation()
        self.reproduce_offspring()
        self.generation_index += 1


if __name__ == "__main__":
    GENERATIONS = 100
    population = Population()
    for generation in range(GENERATIONS):
        population.evolve()
        print(f"gen #{generation} | max fitness: {population.champions[-1].fitness}")

        if population.solved_at is not None or generation == GENERATIONS - 1:
            test_nn = NeuralNetwork(population.champions[-1])
            population.evaluator.log_results(test_nn)
            break
