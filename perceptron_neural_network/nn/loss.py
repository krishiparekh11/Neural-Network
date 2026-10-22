"""
Loss (cost) functions.

cost(A, Y)       -> scalar J averaged over the M samples
derivative(A, Y) -> dL/dA for each sample (the 1/M averaging is applied in BackProp)
"""
from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    name = "base"

    @abstractmethod
    def cost(self, A: np.ndarray, Y: np.ndarray) -> float: ...

    @abstractmethod
    def derivative(self, A: np.ndarray, Y: np.ndarray) -> np.ndarray: ...

    def __repr__(self):
        return self.__class__.__name__


class BinaryCrossEntropy(LossFunction):
    """
    J = -(1/M) sum_p sum_k [ y ln a + (1 - y) ln(1 - a) ]
    For several output neurons, each one is its own binary (one-vs-rest) classifier.
    """
    name = "binary_cross_entropy"

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def cost(self, A, Y):
        M = Y.shape[1]
        A = np.clip(A, self.eps, 1 - self.eps)
        return float(-np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / M)

    def derivative(self, A, Y):
        A = np.clip(A, self.eps, 1 - self.eps)
        return -(Y / A) + (1 - Y) / (1 - A)


class MeanSquaredError(LossFunction):
    """J = (1/M) sum 1/2 (a - y)^2 - for numerical prediction (e.g. house prices)."""
    name = "mse"

    def cost(self, A, Y):
        return float(0.5 * np.sum((A - Y) ** 2) / Y.shape[1])

    def derivative(self, A, Y):
        return A - Y


LOSSES = {cls.name: cls for cls in (BinaryCrossEntropy, MeanSquaredError)}


def get_loss(loss) -> LossFunction:
    if isinstance(loss, LossFunction):
        return loss
    try:
        return LOSSES[loss.lower()]()
    except KeyError:
        raise ValueError(f"Unknown loss '{loss}'. Available: {list(LOSSES)}")
