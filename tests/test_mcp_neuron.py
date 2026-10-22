import pytest

from mcp_neuron import MCPNeuron, Neuron, StepFunction, SumAggregator
from mcp_neuron.logic_gates import AND, OR, not_gate, xor_gate

CASES = [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.mark.parametrize("x", CASES)
def test_and(x):
    assert AND().fire(list(x)) == int(x[0] and x[1])


@pytest.mark.parametrize("x", CASES)
def test_or(x):
    assert OR().fire(list(x)) == int(x[0] or x[1])


@pytest.mark.parametrize("x", CASES)
def test_xor(x):
    assert xor_gate(*x) == x[0] ^ x[1]


def test_not():
    assert not_gate(0) == 1 and not_gate(1) == 0


def test_threshold_is_inclusive():
    assert MCPNeuron(2, theta=2).fire([1, 1]) == 1


def test_rejects_non_boolean_input():
    with pytest.raises(ValueError):
        MCPNeuron(2, theta=1).fire([2, 0])


def test_rejects_wrong_input_count():
    with pytest.raises(ValueError):
        MCPNeuron(3, theta=1).fire([1, 0])


def test_subtractive_inhibition():
    n = MCPNeuron(2, theta=1, n_inhibitory=1, inhibition="subtractive")
    assert n.fire([1, 0], [1]) == 0
    assert n.fire([1, 1], [1]) == 1


def test_generic_neuron_is_pluggable():
    n = Neuron(2, SumAggregator(), StepFunction(0.5))
    assert n.fire([0.3, 0.3]) == 1
