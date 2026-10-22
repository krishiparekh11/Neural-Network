"""Activation functions f(g).

Each activation is a small class with one job: map the aggregated signal g
to an output. New activations (sigmoid, ReLU, spike-with-refractory, ...)
only need to subclass ActivationFunction.
"""
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, g: float) -> float:
        """Return the neuron output for aggregated signal g."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class StepFunction(ActivationFunction):
    """Heaviside step with threshold theta: f(g) = 1 if g >= theta else 0.

    The slides' AND gate (theta = 2) fires when g = 2, so the comparison is
    'greater than or equal'.
    """

    def __init__(self, theta: float):
        self.theta = theta

    def __call__(self, g: float) -> int:
        return 1 if g >= self.theta else 0

    def __repr__(self) -> str:
        return f"StepFunction(theta={self.theta})"
