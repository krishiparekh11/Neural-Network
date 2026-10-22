"""
Parameters of one layer: transmission (weight) matrix W and bias b,
plus their gradients dW, db, and the module for setting their initial values.

Shapes follow the slides:
    W : (N_receiving, N_sending)   e.g. (10, 400) for 10 output neurons, 400 pixels
    b : (N_receiving, 1)           one bias per neuron (column vector)
"""
import numpy as np


class Parameters:
    INITIALIZERS = ("uniform", "normal", "xavier", "he", "zeros")

    def __init__(self, n_sending: int, n_receiving: int, init: str = "xavier",
                 scale: float = 0.01, shared_bias: bool = False,
                 rng: np.random.Generator = None):
        self.n_sending = n_sending
        self.n_receiving = n_receiving
        self.shared_bias = shared_bias           # True = one bias for the whole layer (slides)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.initialize(init, scale)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    # ---- initiation ---------------------------------------------------------
    def initialize(self, init: str = "xavier", scale: float = 0.01):
        """
        Set initial W and b.
          uniform : W ~ U(0, scale)        small random numbers (slides, Ch. 3)
          normal  : W ~ N(0, scale^2)
          xavier  : W ~ N(0, 1/n_sending)  good default for sigmoid / tanh
          he      : W ~ N(0, 2/n_sending)  good default for ReLU
          zeros   : W = 0                  NOT recommended (symmetry, see slides) - for experiments
        Bias always starts at 0.
        """
        shape = (self.n_receiving, self.n_sending)
        n = self.n_sending
        if init == "uniform":
            self.W = self.rng.uniform(0.0, scale, shape)
        elif init == "normal":
            self.W = self.rng.normal(0.0, scale, shape)
        elif init == "xavier":
            self.W = self.rng.normal(0.0, np.sqrt(1.0 / n), shape)
        elif init == "he":
            self.W = self.rng.normal(0.0, np.sqrt(2.0 / n), shape)
        elif init == "zeros":
            self.W = np.zeros(shape)
        else:
            raise ValueError(f"Unknown init '{init}'. Use one of {self.INITIALIZERS}")
        self.b = np.zeros((self.n_receiving, 1))
        self.init = init

    # ---- helpers ------------------------------------------------------------
    def set_gradients(self, dW, db):
        if self.shared_bias:
            db = np.full_like(self.b, db.sum())   # one bias shared by all neurons
        self.dW, self.db = dW, db

    def count(self) -> int:
        return self.W.size + (1 if self.shared_bias else self.b.size)

    def to_dict(self):
        return {"W": self.W.copy(), "b": self.b.copy()}

    def load_dict(self, d):
        assert d["W"].shape == self.W.shape, "Shape mismatch while loading parameters"
        self.W, self.b = d["W"].copy(), d["b"].copy()

    def __repr__(self):
        return f"Parameters(W{self.W.shape}, b{self.b.shape}, init='{self.init}')"
