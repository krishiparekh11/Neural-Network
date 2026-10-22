"""
Activation functions.

Every activation implements forward(Z) and derivative(Z, A).
To add a new one: subclass Activation, implement both methods, and register it
in ACTIVATIONS so it can be selected by name (e.g. activation="tanh").
"""
from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    name = "base"

    @abstractmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        """a = f(z), element-wise."""

    @abstractmethod
    def derivative(self, Z: np.ndarray, A: np.ndarray = None) -> np.ndarray:
        """df/dz, element-wise. A = f(Z) may be passed in to avoid recomputing."""

    def __repr__(self):
        return self.__class__.__name__


class Sigmoid(Activation):
    name = "sigmoid"

    def forward(self, Z):
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))

    def derivative(self, Z, A=None):
        A = self.forward(Z) if A is None else A
        return A * (1.0 - A)                     # sigma'(z) = sigma(z)(1 - sigma(z))


class ReLU(Activation):
    name = "relu"

    def forward(self, Z):
        return np.maximum(Z, 0.0)

    def derivative(self, Z, A=None):
        return (Z >= 0).astype(float)            # 1 for z >= 0, 0 for z < 0 (slides)


class LeakyReLU(Activation):
    name = "leaky_relu"

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, Z):
        return np.where(Z >= 0, Z, self.alpha * Z)

    def derivative(self, Z, A=None):
        return np.where(Z >= 0, 1.0, self.alpha)


class Tanh(Activation):
    name = "tanh"

    def forward(self, Z):
        return np.tanh(Z)

    def derivative(self, Z, A=None):
        A = np.tanh(Z) if A is None else A
        return 1.0 - A ** 2


class Linear(Activation):
    """Identity: used for numerical prediction (linear regression), e.g. house prices."""
    name = "linear"

    def forward(self, Z):
        return Z

    def derivative(self, Z, A=None):
        return np.ones_like(Z)


class Step(Activation):
    """Heaviside step (original perceptron). Not differentiable -> shown for comparison only;
    its 'derivative' is 0 everywhere, so gradient descent cannot train with it."""
    name = "step"

    def forward(self, Z):
        return (Z >= 0).astype(float)

    def derivative(self, Z, A=None):
        return np.zeros_like(Z)


ACTIVATIONS = {cls.name: cls for cls in (Sigmoid, ReLU, LeakyReLU, Tanh, Linear, Step)}


def get_activation(activation) -> Activation:
    """Accepts either a name ("sigmoid") or an Activation instance."""
    if isinstance(activation, Activation):
        return activation
    try:
        return ACTIVATIONS[activation.lower()]()
    except KeyError:
        raise ValueError(f"Unknown activation '{activation}'. Available: {list(ACTIVATIONS)}")
