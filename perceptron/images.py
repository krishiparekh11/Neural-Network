"""Load 20x20 grayscale images as columns of X (400 x M)."""
import os
import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path).convert("L").resize((20, 20))
    return np.asarray(img, dtype=float).reshape(-1) / 255.0


def load_training(folder):
    """folder/<digit>/*.png  ->  X (400, M), Y (10, M) one-hot"""
    X, labels = [], []
    for digit in range(10):
        sub = os.path.join(folder, str(digit))
        for f in sorted(os.listdir(sub)):
            X.append(load_image(os.path.join(sub, f)))
            labels.append(digit)
    labels = np.array(labels)
    Y = np.zeros((10, labels.size))
    Y[labels, np.arange(labels.size)] = 1
    return np.array(X).T, Y


def load_test(folder):
    """folder/*.png  ->  X (400, M), file names"""
    files = sorted(os.listdir(folder))
    return np.array([load_image(os.path.join(folder, f)) for f in files]).T, files
