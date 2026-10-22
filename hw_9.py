#HW - Chapter 9 "Fitting, Bias-Variance, Regularization, and Dropout"
import os
import random

import numpy as np
from PIL import Image, ImageDraw

SEED = 42
IMAGES_PER_CLASS = 100
IMG_SIZE = 64
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
CLASSES = ["circle", "square"]
LABELS = {"circle": 0, "square": 1}

HIDDEN = 32
EPOCHS = 60
LR = 1e-3
BATCH = 16

CONFIGS = {
    "baseline": {},
    "L2 (l2=1e-3)": {"l2": 1e-3},
    "L1 (l1=1e-4)": {"l1": 1e-4},
    "dropout (p=0.5)": {"dropout_rate": 0.5},
    "L2 + dropout": {"l2": 1e-3, "dropout_rate": 0.5},
}


# dataset prototype 

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
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def build_dataset(out_dir):
    
    rng = random.Random(SEED)
    records = []
    for label in CLASSES:
        class_dir = os.path.join(out_dir, "images", label)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            img = draw_image(label, rng)
            path = os.path.join(class_dir, f"{label}_{i:03d}.png")
            img.save(path)
            records.append((path, label))
    return records


def stratified_split(records):
    rng = random.Random(SEED)
    splits = {name: [] for name in SPLIT_RATIOS}
    for label in CLASSES:
        class_records = [r for r in records if r[1] == label]
        rng.shuffle(class_records)
        n = len(class_records)
        n_train = round(n * SPLIT_RATIOS["train"])
        n_val = round(n * SPLIT_RATIOS["validation"])
        splits["train"] += class_records[:n_train]
        splits["validation"] += class_records[n_train:n_train + n_val]
        splits["test"] += class_records[n_train + n_val:]
    for items in splits.values():
        rng.shuffle(items)
    return splits


def print_summary(splits):
    total = sum(len(v) for v in splits.values())
    print("Split summary (stratified)")
    for name, items in splits.items():
        print(f"\n{name.upper()}: {len(items)} images ({len(items) / total:.1%} of total)")
        for label in CLASSES:
            count = sum(1 for _, l in items if l == label)
            print(f"  {label:<8} {count:>3}  ({count / len(items):.1%} of this split)")


def to_arrays(items):
    X, y = [], []
    for path, label in items:
        with Image.open(path) as im:
            x = np.asarray(im.convert("RGB"), dtype=np.float64).reshape(-1) / 255.0
        X.append(x)
        y.append(LABELS[label])
    return np.array(X), np.array(y)


#  neural network with regularization and dropout 

def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def one_hot(y, n_classes):
    return np.eye(n_classes)[y]


class MLP:
    def __init__(self, layer_sizes, l1=0.0, l2=0.0, dropout_rate=0.0, seed=0):
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        self.l1 = l1
        self.l2 = l2
        self.dropout_rate = dropout_rate
        self.rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.W.append(self.rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def _forward(self, X, training):
        acts = [X]    # acts[i] = input to layer i
        zs = []       # pre-activations of hidden layers
        masks = []    # dropout masks of hidden layers (None when unused)
        a = X
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            a = np.maximum(z, 0.0)  # ReLU
            mask = None
            if training and self.dropout_rate > 0.0:
                keep = 1.0 - self.dropout_rate
                mask = (self.rng.random(a.shape) < keep) / keep  # inverted dropout
                a = a * mask
            zs.append(z)
            masks.append(mask)
            acts.append(a)
        logits = a @ self.W[-1] + self.b[-1]
        return acts, zs, masks, logits

    def penalty(self):
        return sum(
            0.5 * self.l2 * np.sum(W ** 2) + self.l1 * np.sum(np.abs(W))
            for W in self.W
        )

    def loss_and_grads(self, X, Y, training=True):
        n = X.shape[0]
        acts, zs, masks, logits = self._forward(X, training)
        probs = softmax(logits)
        data_loss = -np.sum(Y * np.log(probs + 1e-12)) / n
        loss = data_loss + self.penalty()

        gW = [None] * len(self.W)
        gb = [None] * len(self.W)
        delta = (probs - Y) / n  # dLoss/dlogits
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta + self.l2 * self.W[i] + self.l1 * np.sign(self.W[i])
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = delta @ self.W[i].T       # back through layer i
                if masks[i - 1] is not None:
                    delta = delta * masks[i - 1]  # back through dropout
                delta = delta * (zs[i - 1] > 0)   # back through ReLU
        return loss, gW, gb

    def predict_proba(self, X):
        _, _, _, logits = self._forward(X, training=False)
        return softmax(logits)

    def evaluate(self, X, y):
    
        probs = self.predict_proba(X)
        Y = one_hot(y, probs.shape[1])
        loss = -np.sum(Y * np.log(probs + 1e-12)) / len(y) + self.penalty()
        acc = np.mean(probs.argmax(axis=1) == y)
        return loss, acc

    def fit(self, X, y, epochs=50, lr=1e-3, batch_size=16):
        
        Y = one_hot(y, int(y.max()) + 1)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        for _ in range(epochs):
            order = self.rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                _, gW, gb = self.loss_and_grads(X[idx], Y[idx], training=True)
                grads = gW + gb
                t += 1
                for k, (p, g) in enumerate(zip(params, grads)):
                    m[k] = beta1 * m[k] + (1 - beta1) * g
                    v[k] = beta2 * v[k] + (1 - beta2) * g * g
                    m_hat = m[k] / (1 - beta1 ** t)
                    v_hat = v[k] / (1 - beta2 ** t)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)  # in-place update


def gradient_check(eps=1e-5, seed=0):
    
    rng = np.random.default_rng(seed)
    net = MLP([5, 4, 3], l1=0.01, l2=0.1, seed=seed)
    X = rng.normal(size=(6, 5))
    Y = one_hot(rng.integers(0, 3, size=6), 3)
    _, gW, gb = net.loss_and_grads(X, Y, training=False)
    worst = 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for p, g in zip(params, grads):
            for idx in np.ndindex(p.shape):
                orig = p[idx]
                p[idx] = orig + eps
                loss_plus, _, _ = net.loss_and_grads(X, Y, training=False)
                p[idx] = orig - eps
                loss_minus, _, _ = net.loss_and_grads(X, Y, training=False)
                p[idx] = orig
                numeric = (loss_plus - loss_minus) / (2 * eps)
                worst = max(worst, abs(numeric - g[idx]))
    return worst


#  train and compare 
def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    splits = stratified_split(build_dataset(out_dir))
    print_summary(splits)

    print(f"\nGradient check, max abs error: {gradient_check():.2e}\n")

    X_tr, y_tr = to_arrays(splits["train"])
    X_va, y_va = to_arrays(splits["validation"])
    X_te, y_te = to_arrays(splits["test"])

    header = f"{'config':<18}{'train acc':>11}{'val acc':>10}{'test acc':>10}{'mean|W1|':>11}"
    print(header)
    print("-" * len(header))
    for name, cfg in CONFIGS.items():
        net = MLP([X_tr.shape[1], HIDDEN, 2], seed=SEED, **cfg)
        net.fit(X_tr, y_tr, epochs=EPOCHS, lr=LR, batch_size=BATCH)
        _, tr_acc = net.evaluate(X_tr, y_tr)
        _, va_acc = net.evaluate(X_va, y_va)
        _, te_acc = net.evaluate(X_te, y_te)
        mean_w1 = np.mean(np.abs(net.W[0]))
        print(f"{name:<18}{tr_acc:>11.3f}{va_acc:>10.3f}{te_acc:>10.3f}{mean_w1:>11.4f}")


if __name__ == "__main__":
    main()