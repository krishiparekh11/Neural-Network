"""A layer: Z = W A_prev + b,  A = f(Z)."""
from .parameters import Parameters
from .neuron import Neuron


class Layer:
    def __init__(self, n_sending, n_receiving, activation, seed=None):
        self.params = Parameters(n_sending, n_receiving, seed)
        self.activation = activation
        self.neurons = [Neuron(self, k) for k in range(n_receiving)]

    def forward(self, A_prev):
        Z = self.params.W @ A_prev + self.params.b
        return Z, self.activation.forward(Z)
