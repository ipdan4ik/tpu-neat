from dataclasses import dataclass


@dataclass
class Configuration:
    weight_range: tuple[float, float] = (-1, 1)
    max_weight_mutation_delta: float = 0.5
    mutation_add_connection_prob: float = 0.15
    mutation_split_connection_prob: float = 0.25
    mutation_disable_connection_prob: float = 0.1
    mutation_enable_connection_prob: float = 0.1
    mutation_change_weight_prob: float = 0.3
    mutation_reset_weight_prob: float = 0.01

    population_size: int = 100
    target_species: int = 15
    input_nodes: int = 8
    output_nodes: int = 2
    add_bias_node: bool = True
    bias_value: float = 0.5
    max_depth: int = 3

    initial_connection_prob: float = 1
    compatibility_threshold: float = 3.0
    compatibility_threshold_delta: float = 0.4

    distance_excess: float = 1.0
    distance_disjoint: float = 1.0
    distance_weight: float = 0.4

    min_compatibility_threshold: float = 0.1

    stagnation_age: int = 15
    reset_innovations: bool = True
    genome_survival_rate: float = 0.2
    elitism_enabled: bool = True
    max_tournament_champions: int = 3
    random_crossover_weights: bool = False
    prob_reenable_connection: float = 0.15


config = Configuration()
