"""
Image dataset loading. To use a different image set, just point to another folder.

Labeled folder layout:        Unlabeled folder layout:
    train/0/*.png                 test/*.png
    train/1/*.png
    ...
Images are converted to grayscale, resized to 20x20 if needed, scaled to [0, 1],
and inverted automatically if they are dark ink on a light background,
so photos/scans of your own handwriting also work.
"""
import os
import numpy as np
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


class ImageDataset:
    def __init__(self, size=(20, 20), auto_invert=True):
        self.size = size
        self.auto_invert = auto_invert

    def load_image(self, path) -> np.ndarray:
        img = Image.open(path).convert("L")
        if img.size != self.size:
            img = img.resize(self.size, Image.LANCZOS)
        x = np.asarray(img, dtype=float) / 255.0
        if self.auto_invert and x.mean() > 0.5:      # light background -> invert
            x = 1.0 - x
        return x.reshape(-1)                          # flatten row by row -> N = 400

    def load_labeled(self, folder):
        """Returns X (N, M), labels (M,), paths."""
        cols, labels, paths = [], [], []
        for label in sorted(os.listdir(folder)):
            sub = os.path.join(folder, label)
            if not os.path.isdir(sub):
                continue
            for f in sorted(os.listdir(sub)):
                if f.lower().endswith(IMG_EXT):
                    p = os.path.join(sub, f)
                    cols.append(self.load_image(p)); labels.append(int(label)); paths.append(p)
        if not cols:
            raise FileNotFoundError(f"No labeled images found in {folder}")
        return np.stack(cols, axis=1), np.array(labels), paths

    def load_unlabeled(self, folder):
        """Returns X (N, M), paths."""
        paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                       if f.lower().endswith(IMG_EXT))
        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")
        return np.stack([self.load_image(p) for p in paths], axis=1), paths

    @staticmethod
    def one_hot(labels, n_classes=10) -> np.ndarray:
        """labels (M,) -> Y (n_classes, M); column p has a 1 in row labels[p]."""
        Y = np.zeros((n_classes, labels.size))
        Y[labels, np.arange(labels.size)] = 1.0
        return Y

    @staticmethod
    def binary(labels, positive_class) -> np.ndarray:
        """labels (M,) -> Y (1, M): 1 if the image is positive_class, else 0."""
        return (labels == positive_class).astype(float).reshape(1, -1)
