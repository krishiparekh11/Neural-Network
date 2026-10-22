# Perceptron for Handwritten Digits (20x20) - OOP, NumPy

## Quick start
    pip install numpy pillow matplotlib
    python generate_digits.py        # creates data/ (100 labeled training + 30 unlabeled test images)
    python -m tests.test_gradients   # verifies backpropagation numerically
    python main.py                   # trains and tests; settings are in CONFIG at the top

## Class structure (package `nn/`)
| Class | File | Role |
|---|---|---|
| Activation (+ Sigmoid, ReLU, LeakyReLU, Tanh, Linear, Step) | activation.py | f(z) and f'(z); select by name |
| Parameters | parameters.py | transmission matrix W, bias b, gradients; initiation (uniform, normal, xavier, he, zeros) |
| Neuron | neuron.py | view of one neuron (row k of W): z, a, weights as a 20x20 image |
| Layer | layer.py | Parameters + Activation; Z = WA + b, A = f(Z) |
| Model | model.py | list of layers; `Model.perceptron()`, `Model.one_hidden_layer()`; predict, save, load |
| LossFunction (+ BinaryCrossEntropy, MeanSquaredError) | loss.py | cost J and dJ/dA |
| ForwardProp | propagation.py | X -> A, keeps cache |
| BackProp | propagation.py | chain rule -> dW, db for every layer |
| GradDescent | grad_descent.py | W := W - r dW, b := b - r db |
| Training | training.py | epochs loop, stop criterion, history |
| ImageDataset | data.py | loads any image folder, flattens to 400-vectors, one-hot / binary labels |

## Shapes (slide notation)
X (400, M), Y (10, M) one-hot, W (10, 400), b (10, 1), A (10, M).
Each of the 10 output neurons is a logistic-regression perceptron answering "is this digit k?";
the prediction is the neuron with the largest output.
Set `task="binary"` for a single neuron (one digit vs. the rest).

## Changing things
- Image set: point `train_dir` / `test_dir` to other folders (train/<label>/*.png). Scans of real
  handwriting work: images are resized to 20x20 and auto-inverted if dark ink on light paper.
- Activation: `"activation": "tanh"` etc., or write a new `Activation` subclass and add it to `ACTIVATIONS`.
- Initialisation, learning rate, epochs, model type: `CONFIG` in main.py.

## Results (seed 0)
| Setup | Train acc | Test acc (30 images) |
|---|---|---|
| Perceptron, 10 sigmoid outputs | 1.00 | 0.90 |
| Perceptron, single neuron "is it 7?" | 1.00 | 1.00 |
| One hidden layer (32 ReLU) | 1.00 | 0.80 |

## Open issues for class discussion
- 100 training images is very small: 100% training accuracy but lower test accuracy = overfitting.
  With 4,010 parameters and 100 samples, the hidden-layer model (12,000+ parameters) overfits even more.
- Some test outputs have low confidence (e.g. 0.05) yet are the max of 10 - one-vs-rest sigmoids
  do not sum to 1; softmax would be the natural next step for 10 classes.
- Ideas: more / augmented images (shifts, rotations), regularisation, mini-batches, softmax output.
