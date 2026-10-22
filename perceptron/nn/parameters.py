"""Transmission matrix W and bias b of one layer, with their initiation."""
import numpy as np


class Parameters:
    def __init__(self, n_sending, n_receiving, seed=None):
        self.rng = np.random.default_rng(seed)
        self.initialize(n_sending, n_receiving)

    def initialize(self, n_sending, n_receiving):
        # small random weights (not zeros), bias = 0
        self.W = self.rng.normal(0.0, np.sqrt(1.0 / n_sending), (n_receiving, n_sending))
        self.b = np.zeros((n_receiving, 1))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
