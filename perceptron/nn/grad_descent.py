"""Gradient descent: W := W - r dW,  b := b - r db."""


class GradDescent:
    def __init__(self, learning_rate):
        self.r = learning_rate

    def step(self, model):
        for layer in model.layers:
            layer.params.W -= self.r * layer.params.dW
            layer.params.b -= self.r * layer.params.db
