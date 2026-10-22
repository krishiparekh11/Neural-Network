"""Model = list of layers."""
import numpy as np
from .layer import Layer
from .activation import Sigmoid, ReLU
from .forward_prop import ForwardProp


class Model:
    def __init__(self, layers):
        self.layers = layers

    @classmethod
    def perceptron(cls, n_inputs, n_outputs, activation=Sigmoid(), seed=0):
        return cls([Layer(n_inputs, n_outputs, activation, seed)])

    @classmethod
    def one_hidden_layer(cls, n_inputs, n_hidden, n_outputs,
                         hidden_activation=ReLU(), output_activation=Sigmoid(), seed=0):
        return cls([Layer(n_inputs, n_hidden, hidden_activation, seed),
                    Layer(n_hidden, n_outputs, output_activation, seed + 1)])

    def predict(self, X):
        A, _ = ForwardProp(self).run(X)
        return np.argmax(A, axis=0)       # digit whose output neuron is highest
