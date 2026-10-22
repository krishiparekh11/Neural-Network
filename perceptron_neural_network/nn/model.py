"""
Model: an ordered list of layers.

    Model.perceptron(...)        -> single layer (the Perceptron)
    Model.one_hidden_layer(...)  -> hidden layer [1] + output layer [2]
"""
import numpy as np
from .layer import Layer
from .propagation import ForwardProp


class Model:
    def __init__(self, layers, name="model"):
        self.layers = list(layers)
        self.name = name
        for a, b in zip(self.layers, self.layers[1:]):
            assert a.n_receiving == b.n_sending, "Layer sizes do not connect"

    # ---- builders -----------------------------------------------------------
    @classmethod
    def perceptron(cls, n_inputs, n_outputs=1, activation="sigmoid", init="xavier",
                   scale=0.01, shared_bias=False, seed=None):
        rng = np.random.default_rng(seed)
        out = Layer(n_inputs, n_outputs, activation, init, scale, shared_bias, "output", rng)
        return cls([out], name="Perceptron")

    @classmethod
    def one_hidden_layer(cls, n_inputs, n_hidden, n_outputs=1, hidden_activation="relu",
                         output_activation="sigmoid", hidden_init="he", output_init="xavier",
                         shared_bias=False, seed=None):
        rng = np.random.default_rng(seed)
        hidden = Layer(n_inputs, n_hidden, hidden_activation, hidden_init,
                       shared_bias=shared_bias, name="hidden", rng=rng)
        out = Layer(n_hidden, n_outputs, output_activation, output_init,
                    shared_bias=shared_bias, name="output", rng=rng)
        return cls([hidden, out], name="NN with one hidden layer")

    # ---- inference ----------------------------------------------------------
    @property
    def n_outputs(self):
        return self.layers[-1].n_receiving

    def predict_proba(self, X):
        A, _ = ForwardProp(self).run(X)
        return A

    def predict(self, X, threshold=0.5):
        """Single output neuron: 1/0 by threshold. Several outputs: index of the largest."""
        A = self.predict_proba(X)
        if self.n_outputs == 1:
            return (A[0] >= threshold).astype(int)
        return np.argmax(A, axis=0)

    # ---- persistence --------------------------------------------------------
    def save(self, path):
        data = {}
        for i, layer in enumerate(self.layers):
            data[f"W{i}"], data[f"b{i}"] = layer.params.W, layer.params.b
        np.savez(path, **data)

    def load(self, path):
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            layer.params.load_dict({"W": data[f"W{i}"], "b": data[f"b{i}"]})

    def summary(self):
        lines = [f"{self.name}"]
        lines += [f"  [{i + 1}] {layer}" for i, layer in enumerate(self.layers)]
        lines.append(f"  total parameters: {sum(l.params.count() for l in self.layers)}")
        return "\n".join(lines)
