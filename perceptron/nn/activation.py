"""Activation functions. To add a new one, subclass Activation."""
import numpy as np


class Activation:
    def forward(self, Z):
        raise NotImplementedError

    def derivative(self, Z):
        raise NotImplementedError


class Sigmoid(Activation):
    def forward(self, Z):
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))

    def derivative(self, Z):
        s = self.forward(Z)
        return s * (1.0 - s)


class ReLU(Activation):
    def forward(self, Z):
        return np.maximum(Z, 0.0)

    def derivative(self, Z):
        return (Z >= 0).astype(float)
