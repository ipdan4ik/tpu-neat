import random

import numpy as np

from configuration import config
from dataclasses import dataclass


@dataclass
class NodeGene:
    node_id: int
    layer: int


class ConnectionId(tuple):
    def __new__(cls, from_id: int, to_id: int):
        return super(ConnectionId, cls).__new__(cls, (from_id, to_id))

    def __init__(self, from_id: int, to_id: int):
        self.from_id = from_id
        self.to_id = to_id

    def __reduce__(self):
        return self.__class__, (self.from_id, self.to_id)


@dataclass
class ConnectionGene:
    from_node: NodeGene
    to_node: NodeGene
    weight: float
    enabled: bool
    innovation_number: int

    @property
    def connection_id(self):
        return ConnectionId(
            from_id=self.from_node.node_id,
            to_id=self.to_node.node_id,
        )


class InnovationTracker:
    def __init__(self):
        self.current_innovation_number = 0
        self.innovations: dict[ConnectionId, int] = {}

    def get_innovation_number(self, in_node: NodeGene, out_node: NodeGene):
        key = ConnectionId(in_node.node_id, out_node.node_id)
        if key not in self.innovations:
            self.innovations[key] = self.current_innovation_number
            self.current_innovation_number += 1
        return self.innovations[key]

    def reset_innovations(self):
        self.innovations = {}
        self.current_innovation_number = 0


innovation_tracker = InnovationTracker()


class Genome:
    def __init__(self):
        self.connections: dict[ConnectionId, ConnectionGene] = dict()
        self.nodes: list[NodeGene] = list()
        self.fitness: float = 0

    def add_node(self, layer: int) -> NodeGene:
        node = NodeGene(node_id=len(self.nodes), layer=layer)
        self.nodes.append(node)
        return node

    def add_connection(
        self, in_node: NodeGene, out_node: NodeGene, weight: float
    ) -> ConnectionGene:
        connection = ConnectionGene(
            from_node=in_node,
            to_node=out_node,
            weight=weight,
            enabled=True,
            innovation_number=innovation_tracker.get_innovation_number(
                in_node, out_node
            ),
        )
        self.connections[connection.connection_id] = connection
        return connection

    def mutation_add_connection(self):
        potential_connections = []
        for node_i in self.nodes:
            for node_j in self.nodes:
                if (
                    node_i.layer < node_j.layer
                    and ConnectionId(node_i.node_id, node_j.node_id)
                    not in self.connections
                ):
                    potential_connections.append((node_i, node_j))

        if not potential_connections:
            return

        in_node, out_node = random.choice(potential_connections)
        weight = random.uniform(*config.weight_range)
        self.add_connection(in_node, out_node, weight)

    def mutation_split_connection(self):
        possible_connections = []
        for connection in self.connections.values():
            new_layer = (connection.from_node.layer + connection.to_node.layer) // 2
            if new_layer != connection.from_node.layer:
                possible_connections.append(connection)

        if not possible_connections:
            return

        connection = random.choice(possible_connections)
        node = self.add_node(layer=connection.from_node.layer + 1)
        self.add_connection(in_node=connection.from_node, out_node=node, weight=config.weight_range[1])
        self.add_connection(
            in_node=node, out_node=connection.to_node, weight=connection.weight
        )

    def mutation_disable_connection(self):
        possible_connections = list(
            filter(lambda x: x.enabled, self.connections.values())
        )

        if not possible_connections:
            return

        connection = random.choice(list(possible_connections))
        connection.enabled = False

    def mutation_enable_connection(self):
        possible_connections = list(
            filter(lambda x: not x.enabled, self.connections.values())
        )

        if not possible_connections:
            return

        connection = random.choice(list(possible_connections))
        connection.enabled = True

    def mutation_change_weight(self):
        possible_connections = self.connections.values()
        if not possible_connections:
            return

        connection = random.choice(list(possible_connections))

        weight = connection.weight + random.uniform(
            -config.max_weight_mutation_delta,
            config.max_weight_mutation_delta,
        )
        min_weight, max_weight = config.weight_range
        connection.weight = np.clip(weight, min_weight, max_weight)

    def mutation_reset_weight(self):
        possible_connections = self.connections.values()
        if not possible_connections:
            return

        connection = random.choice(list(possible_connections))
        connection.weight = random.uniform(*config.weight_range)

    def mutate(self):
        if random.random() < config.mutation_disable_connection_prob:
            self.mutation_disable_connection()
        if random.random() < config.mutation_enable_connection_prob:
            self.mutation_enable_connection()
        if random.random() < config.mutation_change_weight_prob:
            self.mutation_change_weight()
        if random.random() < config.mutation_reset_weight_prob:
            self.mutation_reset_weight()

        if random.random() < config.mutation_add_connection_prob:
            self.mutation_add_connection()
        if random.random() < config.mutation_split_connection_prob:
            self.mutation_split_connection()
