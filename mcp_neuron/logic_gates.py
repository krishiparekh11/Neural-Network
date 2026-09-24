"""Logic gates built from McCulloch-Pitts neurons (Chapter 1, slides 31-32)."""
from .neuron import MCPNeuron
from .network import NeuronNetwork


def AND(n_inputs: int = 2) -> MCPNeuron:
    """Fires only when all inputs are 1: theta = N."""
    return MCPNeuron(n_inputs, theta=n_inputs, name="AND")


def OR(n_inputs: int = 2) -> MCPNeuron:
    """Fires when at least one input is 1: theta = 1."""
    return MCPNeuron(n_inputs, theta=1, name="OR")


def NOT() -> MCPNeuron:
    """theta = 0 with z wired as an (absolute) inhibitory input.

    A pure excitatory MCP unit cannot invert a signal, so NOT uses the
    inhibitory input from the original 1943 paper. The single excitatory
    input is tied to 0 by the caller; see not_gate().
    """
    return MCPNeuron(1, theta=0, n_inhibitory=1, inhibition="absolute", name="NOT")


def not_gate(z: int) -> int:
    return NOT().fire([0], [z])


def XOR() -> NeuronNetwork:
    """XOR(x1, x2) = AND( NOT(AND(x1, x2)), OR(x1, x2) )  (slide 32)."""
    net = NeuronNetwork(["x1", "x2", "zero"], output_name="xor")
    net.add("and1", AND(), ["x1", "x2"])
    net.add("or1", OR(), ["x1", "x2"])
    net.add("nand", NOT(), ["zero"], inhibitor_sources=["and1"])
    net.add("xor", AND(), ["nand", "or1"])
    return net


def xor_gate(x1: int, x2: int) -> int:
    return XOR().run([x1, x2, 0])
