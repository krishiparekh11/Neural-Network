"""
Layer: a fully connected (dense) layer = Parameters + Activation.

    Z = W A_prev + b        (aggregation)
    A = f(Z)                (activation)
"""
import numpy as np
from .activation import get_activation
from .parameters import Parameters
from .neuron import Neuron


class Layer:
    def __init__(self, n_sending: int, n_receiving: int, activation="sigmoid",
                 init: str = "xavier", scale: float = 0.01, shared_bias: bool = False,
                 name: str = "layer", rng: np.random.Generator = None):
        self.name = name
        self.activation = get_activation(activation)
        self.params = Parameters(n_sending, n_receiving, init, scale, shared_bias, rng)

    @property
    def n_sending(self):
        return self.params.n_sending

    @property
    def n_receiving(self):
        return self.params.n_receiving

    @property
    def neurons(self):
        return [Neuron(self, k) for k in range(self.n_receiving)]

    def aggregate(self, A_prev: np.ndarray) -> np.ndarray:
        return self.params.W @ A_prev + self.params.b   # b (N,1) broadcasts over M columns

    def activate(self, Z: np.ndarray) -> np.ndarray:
        return self.activation.forward(Z)

    def set_activation(self, activation):
        """Swap the activation function (e.g. 'sigmoid' -> 'tanh')."""
        self.activation = get_activation(activation)

    def __repr__(self):
        return (f"Layer('{self.name}', {self.n_sending} -> {self.n_receiving}, "
                f"{self.activation}, params={self.params.count()})")
