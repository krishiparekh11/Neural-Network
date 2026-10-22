"""HW - Chapter 10 “Normalization and Optimization Methods”
Programming Assignment:
Implement input normalization algorithm.
Implement a mini batch approach. """
import math
import random


def normalize_inputs(X, eps=1e-8):
#Input normalization

    m, n = len(X), len(X[0])
    mu = [sum(row[k] for row in X) / m for k in range(n)]
    sigma = [math.sqrt(sum((row[k] - mu[k]) ** 2 for row in X) / m) for k in range(n)]
    X_norm = [
        [(row[k] - mu[k]) / math.sqrt(sigma[k] ** 2 + eps) for k in range(n)]
        for row in X
    ]
    return X_norm, mu, sigma


def make_minibatches(X, Y, batch_size, seed=None):
#Mini-batch approach
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    return [
        ([X[i] for i in idx[s:s + batch_size]], [Y[i] for i in idx[s:s + batch_size]])
        for s in range(0, len(X), batch_size)
    ]


if __name__ == "__main__":
# Quick test: two features on very different scales
    rng = random.Random(0)
    X = [[rng.uniform(1e6, 5e6), rng.uniform(20, 200)] for _ in range(50)]
    Y = [rng.random() for _ in range(50)]

    X_norm, mu, sigma = normalize_inputs(X)
    for k in range(2):
        col = [row[k] for row in X_norm]
        mean = sum(col) / len(col)
        std = math.sqrt(sum((v - mean) ** 2 for v in col) / len(col))
        print(f"feature {k + 1}: mean = {mean:.6f}, std = {std:.6f}")
        assert abs(mean) < 1e-9 and abs(std - 1) < 1e-6

    batches = make_minibatches(X_norm, Y, batch_size=16, seed=1)
    print("mini-batch sizes:", [len(xb) for xb, _ in batches])
    assert sum(len(xb) for xb, _ in batches) == 50
    print("All checks passed.")