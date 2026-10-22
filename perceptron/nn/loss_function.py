"""Cross-entropy loss: J = -(1/M) sum[ y ln a + (1 - y) ln(1 - a) ]."""
import numpy as np


class LossFunction:
    eps = 1e-12

    def cost(self, A, Y):
        A = np.clip(A, self.eps, 1 - self.eps)
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / Y.shape[1]

    def derivative(self, A, Y):
        A = np.clip(A, self.eps, 1 - self.eps)
        return -(Y / A) + (1 - Y) / (1 - A)
