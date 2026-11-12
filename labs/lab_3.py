import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
LAB1_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "lab_1.py"
if not LAB1_PATH.exists():
    sys.exit(f"Lab 1 file not found: {LAB1_PATH}\nPass its path as an argument.")

spec = importlib.util.spec_from_file_location("lab1", LAB1_PATH)
lab1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab1)


X_tr, Y_tr = lab1.make_training_set(n_samples=600, seed=0)
X_va, Y_va = lab1.make_training_set(n_samples=300, seed=7)
X_te, Y_te = lab1.make_training_set(n_samples=300, seed=99)

LR_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
CHECKPOINTS = [1000, 2000, 3000, 5000, 8000, 12000]
INIT_SEEDS = [42, 1, 2]
MAX_EPOCHS = max(CHECKPOINTS)
BASELINE = (0.05, 5000)   # original Lab 1/Lab 2 setting


def train_run(lr, seed):
    """Train once, snapshotting parameters at each checkpoint epoch."""
    params = lab1.initialize_parameters(lab1.LAYER_SIZES, seed=seed)
    snapshots = {}
    for epoch in range(1, MAX_EPOCHS + 1):
        AL, cache = lab1.forward(X_tr, params, lab1.ACTIVATIONS)
        grads = lab1.backward(Y_tr, params, cache, lab1.ACTIVATIONS)
        params = lab1.update_parameters(params, grads, lr)
        if epoch in CHECKPOINTS:
            snapshots[epoch] = {k: v.copy() for k, v in params.items()}
    return snapshots


def val_acc(params):
    return lab1.accuracy(lab1.predict(X_va, params, lab1.ACTIVATIONS), Y_va)


def test_acc(params):
    return lab1.accuracy(lab1.predict(X_te, params, lab1.ACTIVATIONS), Y_te)



val_scores = defaultdict(list)           # (lr, epochs) -> [val acc per seed]
for seed in INIT_SEEDS:
    for lr in LR_GRID:
        snaps = train_run(lr, seed)
        for ep in CHECKPOINTS:
            val_scores[(lr, ep)].append(val_acc(snaps[ep]))

ranked = sorted(
    ((np.mean(v), np.std(v), k) for k, v in val_scores.items()),
    key=lambda x: -x[0],
)
print("Top 10 configs by mean validation accuracy (3 init seeds):")
print(f"{'lr':>7} {'epochs':>7} {'mean_val':>9} {'std':>6}")
for mean, std, (lr, ep) in ranked[:10]:
    print(f"{lr:>7} {ep:>7} {mean:>9.2f} {std:>6.2f}")

best_mean, _, (best_lr, best_ep) = ranked[0]
print(f"\nBest: lr={best_lr}, epochs={best_ep}, mean val acc={best_mean:.2f}%")

best_test = []
for seed in INIT_SEEDS:
    best_test.append(test_acc(train_run(best_lr, seed)[best_ep]))
base_test = test_acc(train_run(*BASELINE[:1], INIT_SEEDS[0])[BASELINE[1]])

print("\nTest accuracy, best config per init seed:", [f"{t:.2f}%" for t in best_test])
print(f"Mean test accuracy, best config: {np.mean(best_test):.2f}%")
print(f"Baseline (lr=0.05, 5000 epochs, seed 42) test accuracy: {base_test:.2f}%")

final_params = train_run(best_lr, INIT_SEEDS[0])[best_ep]
train_final = lab1.accuracy(lab1.predict(X_tr, final_params, lab1.ACTIVATIONS), Y_tr)
print(f"\nFinal model (seed {INIT_SEEDS[0]}): train acc {train_final:.2f}%, "
      f"test acc {test_acc(final_params):.2f}%")

OUT_PATH = HERE / "trained_params_tuned.npz"
np.savez(OUT_PATH, **final_params)
print(f"Saved tuned parameters to {OUT_PATH}")