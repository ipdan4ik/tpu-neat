class NodeGene:
    def __init__(
        self,
        node_id: int,
        layer: int,
    ):
        self.node_id = node_id
        self.layer = layer


class ConnectionGene:
    def __init__(
        self,
        from_node: NodeGene,
        to_node: NodeGene,
        weight: float,
        enabled: bool,
        innovation_number: int,
    ):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number

    @property
    def connection_id(self):
        return self.from_node.node_id, self.to_node.node_id


class Genome:
    def __init__(self):
        self.connections: dict[tuple[int, int], ConnectionGene] = dict()
        self.nodes: list[NodeGene] = list()
        self.fitness: float = 0
