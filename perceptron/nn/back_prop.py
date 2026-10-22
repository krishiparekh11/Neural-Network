"""
Backpropagation:
    dZ[L] = dJ/dA * f'(Z[L])
    dW    = (1/M) dZ A_prev^T,   db = (1/M) sum(dZ)
    dZ[s-1] = (W[s]^T dZ[s]) * f'(Z[s-1])
"""
import numpy as np


class BackProp:
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss

    def run(self, A, cache, Y):
        M = Y.shape[1]
        layers = self.model.layers
        dA = self.loss.derivative(A, Y)
        for s in reversed(range(len(layers))):
            A_prev, Z = cache[s]
            p = layers[s].params
            dZ = dA * layers[s].activation.derivative(Z)
            p.dW = dZ @ A_prev.T / M
            p.db = np.sum(dZ, axis=1, keepdims=True) / M
            dA = p.W.T @ dZ
