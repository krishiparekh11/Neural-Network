"""
Gradient check: compares backprop gradients with numerical derivatives
    dJ/dw ~ (J(w + eps) - J(w - eps)) / (2 eps)
If backprop is correct, the relative error is tiny (< 1e-6).
Run:  python -m tests.test_gradients
"""
import numpy as np
from nn import Model, ForwardProp, BackProp, get_loss


def gradient_check(model, X, Y, loss="binary_cross_entropy", eps=1e-6):
    loss = get_loss(loss)
    A, cache = ForwardProp(model).run(X)
    BackProp(model, loss).run(cache, Y)
    cost = lambda: loss.cost(ForwardProp(model).run(X)[0], Y)
    worst = 0.0
    for layer in model.layers:
        for name in ("W", "b"):
            P = getattr(layer.params, name)
            G = getattr(layer.params, "d" + name)
            for idx in np.ndindex(P.shape):
                old = P[idx]
                P[idx] = old + eps; jp = cost()
                P[idx] = old - eps; jm = cost()
                P[idx] = old
                num = (jp - jm) / (2 * eps)
                worst = max(worst, abs(num - G[idx]) / max(1e-8, abs(num) + abs(G[idx])))
    return worst


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    X = rng.standard_normal((6, 8))
    Y1 = (rng.random((1, 8)) > 0.5).astype(float)
    Y3 = (rng.random((3, 8)) > 0.5).astype(float)
    cases = [
        ("Perceptron, sigmoid, 1 output", Model.perceptron(6, 1, seed=0), Y1, "binary_cross_entropy"),
        ("Perceptron, sigmoid, 3 outputs", Model.perceptron(6, 3, seed=0), Y3, "binary_cross_entropy"),
        ("Perceptron, tanh + MSE", Model.perceptron(6, 1, "tanh", seed=0), Y1, "mse"),
        ("One hidden layer (ReLU/sigmoid)", Model.one_hidden_layer(6, 5, 3, seed=0), Y3, "binary_cross_entropy"),
        ("One hidden layer (tanh/sigmoid)", Model.one_hidden_layer(6, 5, 1, "tanh", seed=0), Y1, "binary_cross_entropy"),
    ]
    for name, model, Y, loss in cases:
        err = gradient_check(model, X, Y, loss)
        print(f"{'PASS' if err < 1e-6 else 'FAIL'}  {name:34s} max relative error = {err:.2e}")
