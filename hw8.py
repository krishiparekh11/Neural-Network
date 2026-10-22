#HW - Chapter 8 "Initialization and Training Sets"
import random

import numpy as np
from PIL import Image, ImageDraw

SEED = 42
IMAGES_PER_CLASS = 100
IMG_SIZE = 64
CLASSES = ["circle", "square"]
LABELS = {"circle": 0, "square": 1}
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}

HIDDEN = 32
EPOCHS = 40
LR = 0.01
BATCH = 16
INIT_METHODS = ["zeros", "small_random", "xavier", "he"]


# dataset 

def draw_image(label, rng):
    background = tuple(int(v) for v in rng.choices(range(0, 80), k=3))
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), background)
    draw = ImageDraw.Draw(img)
    size = rng.randint(18, 36)
    x0 = rng.randint(0, IMG_SIZE - size)
    y0 = rng.randint(0, IMG_SIZE - size)
    box = [x0, y0, x0 + size, y0 + size]
    color = tuple(rng.randint(150, 255) for _ in range(3))
    if label == "circle":
        draw.ellipse(box, fill=color)
    else:
        draw.rectangle(box, fill=color)
    arr = np.array(img, dtype=np.int16)
    noise = np.random.default_rng(rng.randint(0, 10**9)).integers(-10, 11, arr.shape)
    return np.clip(arr + noise, 0, 255).astype(np.uint8)


def make_splits():
    
    rng = random.Random(SEED)
    splits = {name: ([], []) for name in SPLIT_RATIOS}
    for label in CLASSES:
        images = [draw_image(label, rng) for _ in range(IMAGES_PER_CLASS)]
        vectors = [img.reshape(-1).astype(np.float64) / 255.0 for img in images]
        order = list(range(IMAGES_PER_CLASS))
        rng.shuffle(order)
        n_train = round(IMAGES_PER_CLASS * SPLIT_RATIOS["train"])
        n_val = round(IMAGES_PER_CLASS * SPLIT_RATIOS["validation"])
        parts = {
            "train": order[:n_train],
            "validation": order[n_train:n_train + n_val],
            "test": order[n_train + n_val:],
        }
        for name, ids in parts.items():
            for i in ids:
                splits[name][0].append(vectors[i])
                splits[name][1].append(LABELS[label])
    return {name: (np.array(X), np.array(y)) for name, (X, y) in splits.items()}


#  weight initialization 
def init_weights(layer_sizes, method, rng):

    W = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        if method == "zeros":
            w = np.zeros((n_in, n_out))
        elif method == "small_random":
            w = rng.normal(0.0, 0.01, size=(n_in, n_out))
        elif method == "xavier":
            # Glorot: variance keeps activations and gradients balanced
            w = rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), size=(n_in, n_out))
        elif method == "he":
            # He: variance suited to ReLU hidden layers
            w = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        else:
            raise ValueError(f"unknown initialization: {method}")
        W.append(w)
    b = [np.zeros(n) for n in layer_sizes[1:]]
    return W, b


#   network and training 

def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class Network:
    def __init__(self, layer_sizes, init="he", seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = init_weights(layer_sizes, init, rng)

    def forward(self, X):
        acts = [X]    
        zs = []
        a = X
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            zs.append(z)
            a = z if i == last else np.maximum(z, 0.0)  
            acts.append(a)
        return acts, zs

    def loss_and_grads(self, X, Y):
        
        acts, zs = self.forward(X)
        probs = softmax(zs[-1])
        n = X.shape[0]
        loss = -np.sum(Y * np.log(probs + 1e-12)) / n
        gW = [None] * len(self.W)
        gb = [None] * len(self.W)
        delta = (probs - Y) / n
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (zs[i - 1] > 0)  # back through ReLU
        return loss, gW, gb

    def loss(self, X, y):
        probs = softmax(self.forward(X)[1][-1])
        Y = np.eye(2)[y]
        return -np.sum(Y * np.log(probs + 1e-12)) / len(y)

    def predict(self, X):
        return self.forward(X)[1][-1].argmax(axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def fit(self, X, y, epochs, lr, batch_size, rng):
        
        Y = np.eye(2)[y]
        history = []
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                _, gW, gb = self.loss_and_grads(X[idx], Y[idx])
                for i in range(len(self.W)):
                    self.W[i] -= lr * gW[i]
                    self.b[i] -= lr * gb[i]
            history.append(self.loss(X, y))
        return history


#  compare initializations 

def main():
    splits = make_splits()
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["validation"]
    X_te, y_te = splits["test"]
    print(f"Sizes: train {len(X_tr)}, validation {len(X_va)}, test {len(X_te)}\n")

    header = (f"{'init':<14}{'start loss':>12}{'final loss':>12}"
              f"{'train acc':>11}{'val acc':>10}{'test acc':>10}")
    print(header)
    print("-" * len(header))
    for method in INIT_METHODS:
        net = Network([X_tr.shape[1], HIDDEN, 2], init=method, seed=SEED)
        start_loss = net.loss(X_tr, y_tr)
        history = net.fit(X_tr, y_tr, EPOCHS, LR, BATCH, np.random.default_rng(SEED))
        print(f"{method:<14}{start_loss:>12.4f}{history[-1]:>12.4f}"
              f"{net.accuracy(X_tr, y_tr):>11.3f}{net.accuracy(X_va, y_va):>10.3f}"
              f"{net.accuracy(X_te, y_te):>10.3f}")


if __name__ == "__main__":
    main()