"""One neuron k of a layer: z = w . x + b, a = f(z)."""
import numpy as np


class Neuron:
    def __init__(self, layer, k):
        self.layer = layer
        self.k = k

    @property
    def weights(self):
        return self.layer.params.W[self.k]

    @property
    def bias(self):
        return self.layer.params.b[self.k, 0]

    def output(self, x):
        z = self.weights @ np.ravel(x) + self.bias
        return self.layer.activation.forward(z)
