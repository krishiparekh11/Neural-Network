"""
Lab 2 – Training of a Multilayer Neural Network
Programming Assignment
Task 1: Train the Multilayer Neural Network

Train your neural network using your training set both developed in Lab 1.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "trained_params.npz"


def find_lab1_file():
    """Use the path given on the command line, or find a Lab 1 file next to this script."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = HERE / "lab_1.py"
        if not path.exists():
            # Fall back to any file ending in lab_1.py, e.g. 100553e4-lab_1.py
            matches = sorted(HERE.glob("*lab_1.py"))
            if matches:
                path = matches[0]
    if not path.exists():
        sys.exit(f"Lab 1 file not found: {path}\n"
                 f"Put it in the same folder as this script or pass its path as an argument.")
    return path


def load_lab1(path):
    # Load the Lab 1 module by file path (its name starts with a digit,
    # so a normal import statement can't reach it).
    spec = importlib.util.spec_from_file_location("lab1", path)
    lab1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab1)
    return lab1


def main():
    lab1 = load_lab1(find_lab1_file())

    # Training set from Lab 1 (Task 2)
    X, Y = lab1.make_training_set(n_samples=600, seed=0)
    print(f"Training set: {X.shape[1]} objects, {X.shape[0]} features")
    print(f"Inside circle: {int(Y.sum())}   Outside: {int(Y.size - Y.sum())}")
    print(f"Network: {' - '.join(map(str, lab1.LAYER_SIZES[1:]))} "
          f"activations={lab1.ACTIVATIONS}\n")

    # Task 1: train
    params, history = lab1.train(X, Y, lab1.LAYER_SIZES, lab1.ACTIVATIONS)

    # Evaluate
    train_acc = lab1.accuracy(lab1.predict(X, params, lab1.ACTIVATIONS), Y)
    X_test, Y_test = lab1.make_training_set(n_samples=300, seed=99)
    test_acc = lab1.accuracy(lab1.predict(X_test, params, lab1.ACTIVATIONS), Y_test)
    print(f"\nTraining accuracy: {train_acc:.2f}%")
    print(f"Test accuracy:     {test_acc:.2f}%")

    # Save trained parameters so they can be reloaded later
    np.savez(OUT_PATH, **params)
    print(f"Saved parameters to {OUT_PATH}")


if __name__ == "__main__":
    main()