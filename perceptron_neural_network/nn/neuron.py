"""
Neuron: a view of one neuron k inside a Layer.

The network computes whole layers at once with matrices (fast), so a Neuron does
not own separate data. It reads row k of the layer's W and element k of b.
It is useful for inspecting, visualising, or explaining a single neuron:
    z_k = w_k1 x_1 + ... + w_kN x_N + b_k,   a_k = f(z_k)
"""
import numpy as np


class Neuron:
    def __init__(self, layer, index: int):
        self.layer = layer
        self.index = index

    @property
    def weights(self) -> np.ndarray:
        return self.layer.params.W[self.index]          # shape (N_sending,)

    @property
    def bias(self) -> float:
        return float(self.layer.params.b[self.index, 0])

    def aggregate(self, x: np.ndarray) -> float:
        """z = w . x + b for a single input vector x of shape (N,) or (N, 1)."""
        return float(self.weights @ np.ravel(x) + self.bias)

    def activate(self, x: np.ndarray) -> float:
        return float(self.layer.activation.forward(np.array(self.aggregate(x))))

    def weight_image(self, shape=(20, 20)) -> np.ndarray:
        """Weights reshaped as an image - shows which pixels this neuron 'looks for'."""
        return self.weights.reshape(shape)

    def __repr__(self):
        return f"Neuron(layer={self.layer.name}, k={self.index}, bias={self.bias:.4f})"
