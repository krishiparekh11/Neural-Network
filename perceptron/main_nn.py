from nn import Model, Sigmoid, ReLU, LossFunction, GradDescent, Training
from images import load_training, load_test

TRAIN_FOLDER = "data/train"      # change to use a different set of images
TEST_FOLDER = "data/test"
HIDDEN_NEURONS = 32
HIDDEN_ACTIVATION = ReLU()       # hidden layer [1]
OUTPUT_ACTIVATION = Sigmoid()    # output layer [2]
LEARNING_RATE = 0.3
EPOCHS = 1500

# images
X, Y = load_training(TRAIN_FOLDER)
X_test, test_files = load_test(TEST_FOLDER)

# neural network: 400 inputs -> hidden layer [1] -> 10 output neurons [2]
model = Model.one_hidden_layer(n_inputs=X.shape[0], n_hidden=HIDDEN_NEURONS, n_outputs=10,
                               hidden_activation=HIDDEN_ACTIVATION,
                               output_activation=OUTPUT_ACTIVATION)

# train
Training(model, LossFunction(), GradDescent(LEARNING_RATE), EPOCHS, print_every=250).fit(X, Y)

# test
print("\nTraining accuracy:", (model.predict(X) == Y.argmax(axis=0)).mean())
print("\nTest images:")
for f, digit in zip(test_files, model.predict(X_test)):
    print(f"  {f}: {digit}")
