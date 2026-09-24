"""McCulloch-Pitts artificial neuron simulator.

Built as a small framework so later chapters (weights, bias, perceptron
learning, other activation functions, refractory neurons) can plug in
without rewriting the core.
"""
from .activation import ActivationFunction, StepFunction
from .aggregation import Aggregator, SumAggregator
from .neuron import Neuron, MCPNeuron
from .network import NeuronNetwork
from . import logic_gates

__all__ = [
    "ActivationFunction", "StepFunction",
    "Aggregator", "SumAggregator",
    "Neuron", "MCPNeuron",
    "NeuronNetwork", "logic_gates",
]
