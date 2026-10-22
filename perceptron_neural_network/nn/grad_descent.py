"""Gradient descent: W := W - r dW,  b := b - r db  (r = learning rate)."""


class GradDescent:
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate

    def step(self, model):
        r = self.learning_rate
        for layer in model.layers:
            p = layer.params
            p.W -= r * p.dW
            p.b -= r * p.db

    def __repr__(self):
        return f"GradDescent(r={self.learning_rate})"
