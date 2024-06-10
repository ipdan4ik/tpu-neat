from enum import Enum, member
import numpy as np


def hyperbolic_tangent(x):
    return np.tanh(x)


def sigmoid(x):
    return (np.tanh(x / 2.0) + 1.0) / 2.0


def sigmoid_v2(x):
    return 1 / (1 + np.exp(-4.9 * x))


def relu(x):
    return np.maximum(0, x)


class ActivationF(Enum):
    TANH = member(hyperbolic_tangent)
    SIGMOID = member(sigmoid)
    RELU = member(relu)
    SIGMOID_V2 = member(sigmoid_v2)
