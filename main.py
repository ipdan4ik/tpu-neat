class NodeGene:
    def __init__(
        self,
        node_id: int,
        layer: int,
    ):
        self.node_id = node_id
        self.layer = layer


class ConnectionId(tuple):
    def __new__(cls, from_id: int, to_id: int):
        return super(ConnectionId, cls).__new__(cls, (from_id, to_id))

    def __init__(self, from_id: int, to_id: int):
        self.from_id = from_id
        self.to_id = to_id

    def __reduce__(self):
        return self.__class__, (self.from_id, self.to_id)


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
        return ConnectionId(
            from_id=self.from_node.node_id,
            to_id=self.to_node.node_id,
        )


class InnovationTracker:
    def __init__(self):
        self.current_innovation_number = 0
        self.innovations = {}

    def get_innovation_number(self, in_node: NodeGene, out_node: NodeGene):
        key = (in_node.node_id, out_node.node_id)
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
