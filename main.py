from dataclasses import dataclass

from activation import ActivationF


@dataclass
class NodeGene:
    node_id: int
    layer: int
    activation_f: ActivationF = ActivationF.TANH


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


