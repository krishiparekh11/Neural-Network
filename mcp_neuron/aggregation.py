"""Aggregation functions g(X).

The McCulloch-Pitts neuron uses a plain sum of its inputs. The perceptron
(Chapter 2) needs a weighted sum plus bias; that becomes another subclass
here rather than a change to the neuron.
"""
from abc import ABC, abstractmethod
from typing import Sequence


class Aggregator(ABC):
    @abstractmethod
    def __call__(self, inputs: Sequence[float]) -> float:
        """Combine the input vector into a single signal g."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class SumAggregator(Aggregator):
    """g(X) = x1 + x2 + ... + xN  (the original MCP aggregation)."""

    def __call__(self, inputs: Sequence[float]) -> float:
        return sum(inputs)
