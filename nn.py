from configuration import config

from genome import Genome
import matplotlib.pyplot as plt
import networkx as nx


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

    def visualize(self, show_weights=True):
        graph = nx.DiGraph()

        pos = {}
        labels = {}

        layers = sorted(set(node.layer for node in self.nodes.values()))
        layer_nodes = {
            layer: [node for node in self.nodes.values() if node.layer == layer]
            for layer in layers
        }

        max_nodes_in_layer = max(len(nodes) for nodes in layer_nodes.values())
        horizontal_spacing = 2
        vertical_spacing = 2

        for layer in layers:
            nodes = layer_nodes[layer]
            num_nodes = len(nodes)
            y_offset = (max_nodes_in_layer - num_nodes) * vertical_spacing / 2
            for i, node in enumerate(nodes):
                pos[node.node_id] = (
                    layer * horizontal_spacing,
                    i * vertical_spacing + y_offset,
                )
                if layer == 0 and i == len(nodes) - 1 and config.add_bias_node:
                    labels[node.node_id] = "bias"
                else:
                    labels[node.node_id] = f"{node.node_id} ({node.layer})"

        edge_labels = {}
        for conn in self.connections:
            edge_color = "black" if conn.weight > 0 else "gray"
            graph.add_edge(conn.from_node.node_id, conn.to_node.node_id, color=edge_color)
            if show_weights:
                edge_labels[(conn.from_node.node_id, conn.to_node.node_id)] = (
                    f"{conn.weight:.2f}"
                )

        edges = graph.edges()
        colors = [graph[u][v]["color"] for u, v in edges]

        plt.figure(figsize=(12, 8))
        nx.draw(
            graph,
            pos,
            with_labels=True,
            labels=labels,
            node_size=3000,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
            arrowsize=20,
            edge_color=colors,
        )
        if show_weights:
            nx.draw_networkx_edge_labels(
                graph, pos, edge_labels=edge_labels, font_color="blue"
            )

        plt.title("Neural Network Visualization")
        plt.show()


