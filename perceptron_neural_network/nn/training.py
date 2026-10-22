"""
Training: the supervised training loop.

    repeat:
        forward propagation  -> A
        loss                 -> J(A, Y)
        stop if J <= target_cost
        backpropagation      -> dW, db
        gradient descent     -> W, b updated
"""
import numpy as np
from .propagation import ForwardProp, BackProp
from .loss import get_loss


class Training:
    def __init__(self, model, loss="binary_cross_entropy", optimizer=None,
                 epochs=1000, target_cost=None, print_every=100):
        from .grad_descent import GradDescent
        self.model = model
        self.loss = get_loss(loss)
        self.optimizer = optimizer or GradDescent(0.1)
        self.epochs = epochs
        self.target_cost = target_cost
        self.print_every = print_every
        self.forward = ForwardProp(model)
        self.backward = BackProp(model, self.loss)
        self.history = {"cost": [], "train_acc": [], "val_acc": []}

    @staticmethod
    def accuracy(model, X, labels):
        return float(np.mean(model.predict(X) == labels))

    def fit(self, X, Y, labels=None, X_val=None, labels_val=None):
        """
        X: (N, M) inputs, Y: (n_outputs, M) targets.
        labels (M,): class labels used only to report accuracy (optional).
        """
        for epoch in range(1, self.epochs + 1):
            A, cache = self.forward.run(X)
            J = self.loss.cost(A, Y)
            self.history["cost"].append(J)

            if labels is not None:
                self.history["train_acc"].append(self.accuracy(self.model, X, labels))
            if X_val is not None:
                self.history["val_acc"].append(self.accuracy(self.model, X_val, labels_val))

            if not np.isfinite(J):
                print(f"epoch {epoch}: cost is {J} - training diverged. Try a smaller learning rate.")
                break
            if self.target_cost is not None and J <= self.target_cost:
                print(f"epoch {epoch}: target cost {self.target_cost} reached (J = {J:.5f})")
                break

            self.backward.run(cache, Y)
            self.optimizer.step(self.model)

            if self.print_every and epoch % self.print_every == 0:
                msg = f"epoch {epoch:5d}  J = {J:.5f}"
                if labels is not None:
                    msg += f"  train acc = {self.history['train_acc'][-1]:.3f}"
                if X_val is not None:
                    msg += f"  val acc = {self.history['val_acc'][-1]:.3f}"
                print(msg)
        return self.history
