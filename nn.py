from configuration import config

from genome import Genome


class NeuralNetwork:
    def __init__(self, genome: Genome):
        self.genome = genome
        self.nodes = {node.node_id: node for node in genome.nodes}
        self.connections = [
            conn for conn in genome.connections.values() if conn.enabled
        ]

    def feed(self, inputs: list[float]) -> list[float]:
        node_values = {node.node_id: 0.0 for node in self.nodes.values()}

        input_nodes = [node for node in self.nodes.values() if node.layer == 0]
        for i, input_value in enumerate(inputs):
            node_values[input_nodes[i].node_id] = input_value

        if config.add_bias_node:
            node_values[input_nodes[-1].node_id] = config.bias_value

        layers = sorted(set(node.layer for node in self.nodes.values()))
        for layer in layers[1:]:
            for node in [n for n in self.nodes.values() if n.layer == layer]:
                input_sum = sum(
                    node_values[conn.from_node.node_id] * conn.weight
                    for conn in self.connections
                    if conn.to_node.node_id == node.node_id
                )
                node_values[node.node_id] = node.activation_f.value(
                    input_sum
                )

        output_nodes = [
            node for node in self.nodes.values() if node.layer == max(layers)
        ]

        return [node_values[node.node_id] for node in output_nodes]
