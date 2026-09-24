"""Tiny feed-forward composition of neurons.

Each node names its neuron and which signals feed its excitatory and
inhibitory inputs. Signals are either external inputs or outputs of earlier
nodes. Enough to build XOR from MCP neurons, and a starting point for
multilayer networks later in the course.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class _Node:
    name: str
    neuron: object
    sources: List[str]
    inhibitor_sources: List[str] = field(default_factory=list)


class NeuronNetwork:
    def __init__(self, input_names: Sequence[str], output_name: str):
        self.input_names = list(input_names)
        self.output_name = output_name
        self._nodes: List[_Node] = []

    def add(self, name: str, neuron, sources: Sequence[str],
            inhibitor_sources: Sequence[str] = ()) -> "NeuronNetwork":
        known = set(self.input_names) | {n.name for n in self._nodes}
        for s in list(sources) + list(inhibitor_sources):
            if s not in known:
                raise ValueError(f"Unknown signal {s!r} feeding node {name!r}")
        self._nodes.append(_Node(name, neuron, list(sources), list(inhibitor_sources)))
        return self

    def run(self, inputs: Sequence[int], trace: bool = False):
        if len(inputs) != len(self.input_names):
            raise ValueError("Wrong number of network inputs")
        signals: Dict[str, int] = dict(zip(self.input_names, inputs))
        for node in self._nodes:
            x = [signals[s] for s in node.sources]
            if node.inhibitor_sources:
                h = [signals[s] for s in node.inhibitor_sources]
                signals[node.name] = node.neuron.fire(x, h)
            else:
                signals[node.name] = node.neuron.fire(x)
        out = signals[self.output_name]
        return (out, signals) if trace else out

    __call__ = run
