"""
Train and test the Perceptron on 20x20 handwritten digits.

Everything you may want to change is in CONFIG:
  - image folders (swap in a different image set)
  - activation function, initialisation, learning rate, epochs
  - task: "multiclass" (10 output neurons, one per digit) or "binary" (one neuron: digit X or not)
  - model: "perceptron" or "one_hidden_layer"

Run:  python main.py
"""
import csv
import os
import numpy as np
from nn import Model, Training, GradDescent, ImageDataset

CONFIG = {
    "train_dir": "data/train",
    "test_dir": "data/test",
    "answer_key": "data/test_answer_key.csv",   # optional, only to score the test run
    "image_size": (20, 20),

    "task": "multiclass",          # "multiclass" or "binary"
    "positive_digit": 7,           # used when task == "binary"

    "model": "perceptron",         # "perceptron" or "one_hidden_layer"
    "hidden_neurons": 32,
    "activation": "sigmoid",       # output activation: sigmoid, tanh, relu, linear ...
    "hidden_activation": "relu",
    "init": "xavier",              # uniform, normal, xavier, he, zeros
    "loss": "binary_cross_entropy",

    "learning_rate": 0.5,
    "epochs": 1500,
    "target_cost": 0.01,
    "print_every": 250,
    "seed": 0,
}


def build_model(cfg, n_inputs, n_outputs):
    if cfg["model"] == "perceptron":
        return Model.perceptron(n_inputs, n_outputs, cfg["activation"], cfg["init"], seed=cfg["seed"])
    return Model.one_hidden_layer(n_inputs, cfg["hidden_neurons"], n_outputs,
                                  cfg["hidden_activation"], cfg["activation"], seed=cfg["seed"])


def read_answer_key(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return {row["file"]: int(row["label"]) for row in csv.DictReader(f)}


def save_plots(history, model, cfg, out="results"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history["cost"]); ax[0].set_title("Loss J(W, b)"); ax[0].set_xlabel("epoch")
    ax[1].plot(history["train_acc"]); ax[1].set_title("Training accuracy"); ax[1].set_xlabel("epoch")
    ax[1].set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(f"{out}/training_curves.png", dpi=110); plt.close(fig)

    # What each output neuron learned: its 400 weights shown as a 20x20 image
    if cfg["model"] == "perceptron":
        neurons = model.layers[-1].neurons
        fig, axes = plt.subplots(1, len(neurons), figsize=(1.4 * len(neurons), 1.8), squeeze=False)
        for a, n in zip(axes[0], neurons):
            a.imshow(n.weight_image(cfg["image_size"]), cmap="coolwarm"); a.axis("off")
            a.set_title(str(n.index) if len(neurons) > 1 else f"is {cfg['positive_digit']}?")
        fig.suptitle("Output neuron weights (red = pixel votes for, blue = against)", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{out}/neuron_weights.png", dpi=110); plt.close(fig)


def main(cfg=CONFIG):
    data = ImageDataset(cfg["image_size"])
    X, labels, _ = data.load_labeled(cfg["train_dir"])
    X_test, test_paths = data.load_unlabeled(cfg["test_dir"])

    if cfg["task"] == "binary":
        Y, n_out = data.binary(labels, cfg["positive_digit"]), 1
        target_labels = (labels == cfg["positive_digit"]).astype(int)
    else:
        Y, n_out = data.one_hot(labels, 10), 10
        target_labels = labels

    print(f"Training set: X {X.shape}, Y {Y.shape}   Test set: {X_test.shape[1]} unlabeled images")
    model = build_model(cfg, X.shape[0], n_out)
    print(model.summary(), "\n")

    trainer = Training(model, cfg["loss"], GradDescent(cfg["learning_rate"]),
                       cfg["epochs"], cfg["target_cost"], cfg["print_every"])
    history = trainer.fit(X, Y, target_labels)
    print(f"\nFinal training accuracy: {Training.accuracy(model, X, target_labels):.3f}")

    # ---- test on the unlabeled images ----
    P = model.predict_proba(X_test)
    pred = model.predict(X_test)
    key = read_answer_key(cfg["answer_key"])
    print("\nTest images:")
    correct = 0
    for i, path in enumerate(test_paths):
        name = os.path.basename(path)
        if n_out == 1:
            shown = f"P(is {cfg['positive_digit']}) = {P[0, i]:.3f} -> {'yes' if pred[i] else 'no'}"
        else:
            shown = f"predicted {pred[i]}  (confidence {P[pred[i], i]:.3f})"
        truth = ""
        if key and name in key:
            true = key[name] if n_out > 1 else int(key[name] == cfg["positive_digit"])
            ok = true == pred[i]
            correct += ok
            truth = f"   true {key[name]}  {'OK' if ok else 'WRONG'}"
        print(f"  {name}: {shown}{truth}")
    if key:
        print(f"\nTest accuracy (checked against answer key): {correct}/{len(test_paths)}"
              f" = {correct / len(test_paths):.3f}")

    os.makedirs("results", exist_ok=True)
    model.save("results/trained_parameters.npz")
    save_plots(history, model, cfg)
    print("Saved parameters and plots to results/")
    return model, history


if __name__ == "__main__":
    main()
