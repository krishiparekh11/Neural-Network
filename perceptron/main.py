from nn import Model, Sigmoid, LossFunction, GradDescent, Training
from images import load_training, load_test

TRAIN_FOLDER = "data/train"     # change to use a different set of images
TEST_FOLDER = "data/test"
ACTIVATION = Sigmoid()          # change to use a different activation function
LEARNING_RATE = 0.5
EPOCHS = 1500

# images
X, Y = load_training(TRAIN_FOLDER)
X_test, test_files = load_test(TEST_FOLDER)

# perceptron: 400 inputs -> 10 output neurons (one per digit), parameters initiated inside
model = Model.perceptron(n_inputs=X.shape[0], n_outputs=10, activation=ACTIVATION)

# train
Training(model, LossFunction(), GradDescent(LEARNING_RATE), EPOCHS, print_every=250).fit(X, Y)

# test
print("\nTraining accuracy:", (model.predict(X) == Y.argmax(axis=0)).mean())
print("\nTest images:")
for f, digit in zip(test_files, model.predict(X_test)):
    print(f"  {f}: {digit}")
