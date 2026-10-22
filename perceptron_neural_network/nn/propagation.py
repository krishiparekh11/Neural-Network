"""
Forward propagation and backpropagation, written for any number of layers,
so the same code runs the Perceptron (1 layer) and the network with one hidden layer (2 layers).
"""
import numpy as np
from .activation import Sigmoid
from .loss import BinaryCrossEntropy


class ForwardProp:
    """X -> Z[1], A[1] -> ... -> Z[L], A[L]. Stores a cache for backpropagation."""

    def __init__(self, model):
        self.model = model

    def run(self, X: np.ndarray):
        cache = []
        A = X
        for layer in self.model.layers:
            Z = layer.aggregate(A)
            A_next = layer.activate(Z)
            cache.append({"A_prev": A, "Z": Z, "A": A_next})
            A = A_next
        return A, cache


class BackProp:
    """
    Chain rule from the loss back to every W and b.

      dZ[L] = dL/dA[L] * f'(Z[L])          (= A - Y for sigmoid + cross-entropy)
      dW[s] = (1/M) dZ[s] A[s-1]^T
      db[s] = (1/M) dZ[s] E_M              (sum over samples)
      dZ[s-1] = (W[s]^T dZ[s]) * f'(Z[s-1])   (* = Hadamard product)
    """

    def __init__(self, model, loss):
        self.model = model
        self.loss = loss

    def run(self, cache, Y: np.ndarray):
        M = Y.shape[1]
        layers = self.model.layers
        A_L = cache[-1]["A"]

        # Output layer
        last = layers[-1]
        if isinstance(last.activation, Sigmoid) and isinstance(self.loss, BinaryCrossEntropy):
            dZ = A_L - Y              # exact simplification, numerically stable
        else:
            dZ = self.loss.derivative(A_L, Y) * last.activation.derivative(cache[-1]["Z"], A_L)

        # Walk backwards through the layers
        for s in reversed(range(len(layers))):
            layer, c = layers[s], cache[s]
            dW = (1.0 / M) * dZ @ c["A_prev"].T
            db = (1.0 / M) * np.sum(dZ, axis=1, keepdims=True)
            layer.params.set_gradients(dW, db)
            if s > 0:
                prev = layers[s - 1]
                dA_prev = layer.params.W.T @ dZ
                dZ = dA_prev * prev.activation.derivative(cache[s - 1]["Z"], cache[s - 1]["A"])
