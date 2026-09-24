"""Neuron classes.

Neuron      - generic unit: y = f(g(X)), with pluggable g and f.
MCPNeuron   - McCulloch-Pitts neuron: Boolean inputs, sum aggregation,
              step activation with threshold theta, optional inhibitory
              inputs (off by default, since the course's MCP model has none).
"""
from typing import Optional, Sequence

from .activation import ActivationFunction, StepFunction
from .aggregation import Aggregator, SumAggregator


class Neuron:
    """Generic artificial neuron: output = activation(aggregator(inputs))."""

    def __init__(self, n_inputs: int, aggregator: Aggregator,
                 activation: ActivationFunction, name: str = "neuron"):
        if n_inputs < 1:
            raise ValueError("A neuron needs at least one input.")
        self.n_inputs = n_inputs
        self.aggregator = aggregator
        self.activation = activation
        self.name = name

    def _validate(self, inputs: Sequence[float]) -> None:
        if len(inputs) != self.n_inputs:
            raise ValueError(
                f"{self.name}: expected {self.n_inputs} inputs, got {len(inputs)}")

    def aggregate(self, inputs: Sequence[float]) -> float:
        self._validate(inputs)
        return self.aggregator(inputs)

    def fire(self, inputs: Sequence[float]):
        """Compute the neuron's output for one input pattern."""
        return self.activation(self.aggregate(inputs))

    __call__ = fire

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name={self.name!r}, "
                f"n_inputs={self.n_inputs}, {self.aggregator!r}, {self.activation!r})")


class MCPNeuron(Neuron):
    """McCulloch-Pitts neuron (1943).

    y = 1 if (x1 + ... + xN) >= theta else 0,  with every xk in {0, 1}.

    Optional inhibitory inputs H = {h1..hK} (extension, not part of the
    course's MCP model): in 'absolute' mode (as in the 1943 paper) any active
    inhibitor forces y = 0; in 'subtractive' mode g = sum(X) - sum(H).
    """

    INHIBITION_MODES = ("absolute", "subtractive")

    def __init__(self, n_inputs: int, theta: float, n_inhibitory: int = 0,
                 inhibition: str = "absolute", name: str = "mcp"):
        super().__init__(n_inputs, SumAggregator(), StepFunction(theta), name)
        if inhibition not in self.INHIBITION_MODES:
            raise ValueError(f"inhibition must be one of {self.INHIBITION_MODES}")
        self.n_inhibitory = n_inhibitory
        self.inhibition = inhibition

    @property
    def theta(self) -> float:
        return self.activation.theta

    @staticmethod
    def _check_boolean(values: Sequence[int], label: str) -> None:
        for v in values:
            if v not in (0, 1):
                raise ValueError(f"MCP {label} must be Boolean (0 or 1); got {v!r}")

    def fire(self, inputs: Sequence[int],
             inhibitors: Optional[Sequence[int]] = None) -> int:
        self._check_boolean(inputs, "inputs")
        inhibitors = list(inhibitors or [])
        if len(inhibitors) != self.n_inhibitory:
            raise ValueError(
                f"{self.name}: expected {self.n_inhibitory} inhibitory inputs, "
                f"got {len(inhibitors)}")
        self._check_boolean(inhibitors, "inhibitory inputs")

        g = self.aggregate(inputs)
        if inhibitors:
            if self.inhibition == "absolute" and any(inhibitors):
                return 0
            if self.inhibition == "subtractive":
                g -= sum(inhibitors)
        return self.activation(g)

    __call__ = fire
