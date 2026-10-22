"""
Lab 1 - Develop a Multilayer Neural Network
Programming Assignment
Task 1: Develop a Multilayer (Deep) Neural Network

Develop a multilayer (deep) neural networks for binary classification problem according to the following specification.

Model: (10/ReLU) – (8/ReLU)² – (4/ReLU) – (1/Sigmoid)
The specification above means the following:
layer 1 with 10 neurons and ReLU activation function
layers 2 and 3 (2 layers shown as power 2 in the spec) with 8 neurons per each layer and ReLU activation function
layer 4 with 4 neurons and ReLU activation function
one (last) layer 5 with one neuron and Sigmoid activation function
Task 2: Develop a training set
Choose objects to train the network and develop a training set

Task 3; GitHub
If you do not have your GitHub repository, then create it. If you are not clear how to do it, ask the TA.
"""

import numpy as np

# Network specification

INPUT_FEATURES = 2
LAYER_SIZES = [INPUT_FEATURES, 10, 8, 8, 4, 1]          # input + 5 layers
ACTIVATIONS = ["relu", "relu", "relu", "relu", "sigmoid"]

LEARNING_RATE = 0.05
EPOCHS = 5000
SEED = 42

# Task 2: training set

def make_training_set(n_samples=600, seed=0):
    """Return X with shape (2, m) and Y with shape (1, m)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
    Y = (np.sum(X ** 2, axis=1) < 1.0).astype(float).reshape(1, -1)
    return X.T, Y


# Activation functions

def activate(z, name):
    if name == "relu":
        return np.maximum(0, z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(name)


def activate_derivative(z, a, name):
    """Derivative of the activation, given pre-activation z and output a."""
    if name == "relu":
        return (z > 0).astype(float)
    if name == "sigmoid":
        return a * (1.0 - a)
    raise ValueError(name)

# Parameters

def initialize_parameters(layer_sizes, seed=SEED):
    """He initialization for weights, zeros for biases."""
    rng = np.random.default_rng(seed)
    params = {}
    for l in range(1, len(layer_sizes)):
        params[f"W{l}"] = rng.standard_normal((layer_sizes[l], layer_sizes[l - 1])) \
                          * np.sqrt(2.0 / layer_sizes[l - 1])
        params[f"b{l}"] = np.zeros((layer_sizes[l], 1))
    return params

# Forward and backward propagation

def forward(X, params, activations):
    cache = {"A0": X}
    A = X
    L = len(activations)
    for l in range(1, L + 1):
        Z = params[f"W{l}"] @ A + params[f"b{l}"]
        A = activate(Z, activations[l - 1])
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
    return A, cache


def compute_cost(AL, Y):
    """Binary cross-entropy."""
    m = Y.shape[1]
    eps = 1e-12
    return -np.sum(Y * np.log(AL + eps) + (1 - Y) * np.log(1 - AL + eps)) / m


def backward(Y, params, cache, activations):
    grads = {}
    m = Y.shape[1]
    L = len(activations)
    AL = cache[f"A{L}"]
    # For sigmoid output with cross-entropy, dZ = A - Y
    dA = -(Y / (AL + 1e-12)) + (1 - Y) / (1 - AL + 1e-12)
    for l in range(L, 0, -1):
        Z = cache[f"Z{l}"]
        A = cache[f"A{l}"]
        A_prev = cache[f"A{l - 1}"]
        if l == L:
            dZ = AL - Y
        else:
            dZ = dA * activate_derivative(Z, A, activations[l - 1])
        grads[f"dW{l}"] = dZ @ A_prev.T / m
        grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
        dA = params[f"W{l}"].T @ dZ
    return grads


def update_parameters(params, grads, lr):
    L = len([k for k in params if k.startswith("W")])
    for l in range(1, L + 1):
        params[f"W{l}"] -= lr * grads[f"dW{l}"]
        params[f"b{l}"] -= lr * grads[f"db{l}"]
    return params

# Training and evaluation

def train(X, Y, layer_sizes, activations, lr=LEARNING_RATE, epochs=EPOCHS):
    params = initialize_parameters(layer_sizes)
    history = []
    for epoch in range(1, epochs + 1):
        AL, cache = forward(X, params, activations)
        cost = compute_cost(AL, Y)
        grads = backward(Y, params, cache, activations)
        params = update_parameters(params, grads, lr)
        if epoch % 500 == 0 or epoch == 1:
            history.append((epoch, cost))
            print(f"Epoch {epoch:5d}   cost = {cost:.4f}")
    return params, history


def predict(X, params, activations):
    AL, _ = forward(X, params, activations)
    return (AL > 0.5).astype(int)


def accuracy(predictions, Y):
    return np.mean(predictions == Y.astype(int)) * 100


def main():
    X, Y = make_training_set()
    print(f"Training set: {X.shape[1]} objects, {X.shape[0]} features each")
    print(f"Positive (inside circle): {int(Y.sum())}   Negative: {int(Y.size - Y.sum())}")
    print(f"Network: {' - '.join(str(n) for n in LAYER_SIZES[1:])} "
          f"with activations {ACTIVATIONS}\n")

    params, _ = train(X, Y, LAYER_SIZES, ACTIVATIONS)

    preds = predict(X, params, ACTIVATIONS)
    print(f"\nTraining accuracy: {accuracy(preds, Y):.2f}%")

    # Test on new points the network has not seen
    X_test, Y_test = make_training_set(n_samples=300, seed=99)
    test_preds = predict(X_test, params, ACTIVATIONS)
    print(f"Test accuracy:     {accuracy(test_preds, Y_test):.2f}%")


if __name__ == "__main__":
    main()