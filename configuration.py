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


config = Configuration()
