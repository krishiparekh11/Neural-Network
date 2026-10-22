"""Training loop: forward prop -> loss -> backprop -> gradient descent."""
from .forward_prop import ForwardProp
from .back_prop import BackProp


class Training:
    def __init__(self, model, loss, optimizer, epochs, print_every=100):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.print_every = print_every

    def fit(self, X, Y):
        forward = ForwardProp(self.model)
        backward = BackProp(self.model, self.loss)
        for epoch in range(1, self.epochs + 1):
            A, cache = forward.run(X)
            J = self.loss.cost(A, Y)
            backward.run(A, cache, Y)
            self.optimizer.step(self.model)
            if epoch % self.print_every == 0:
                print(f"epoch {epoch:5d}   J = {J:.5f}")
