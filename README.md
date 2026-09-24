# McCulloch-Pitts Neuron Simulator

Chapter 1 programming assignment: a simulation of the McCulloch and Pitts (1943) artificial neuron.

The neuron takes Boolean inputs x1..xN, aggregates them as g = x1 + ... + xN, and fires y = 1 if g >= θ, otherwise y = 0.

## Run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                   # truth tables for AND, OR, NOT, XOR + interactive neuron
pytest                           # tests
```

## Usage

```python
from mcp_neuron import MCPNeuron
n = MCPNeuron(n_inputs=3, theta=2)
n.fire([1, 0, 1])   # -> 1
```

## Design (built for later chapters)

| Module | Role | How it extends |
|---|---|---|
| `activation.py` | `ActivationFunction` base + `StepFunction(θ)` | add sigmoid, ReLU, spike/refractory activations as subclasses |
| `aggregation.py` | `Aggregator` base + `SumAggregator` | add `WeightedSumAggregator(w, b)` for the perceptron |
| `neuron.py` | generic `Neuron` (g, f pluggable) + `MCPNeuron` | perceptron = `Neuron` with a weighted aggregator and a training method |
| `network.py` | `NeuronNetwork` feed-forward composition | basis for multilayer networks |
| `logic_gates.py` | AND, OR, NOT, XOR from MCP neurons | examples / regression tests |

Inhibitory inputs are an optional extension (off by default) since they are not part of the MCP model as taught; NOT uses one, as in the original 1943 paper.
