import numpy as np
from typing import List, Tuple, Union, Optional


# ==============================================================================
# PART 1: ACTIVATION FUNCTIONS
# ==============================================================================

class ActivationFunctions:
    """
    Class containing various activation functions and their derivatives.

    Activation functions are essential for introducing non-linearity into
    neural networks, allowing them to learn complex patterns.
    """

    @staticmethod
    def linear(Z: np.ndarray) -> np.ndarray:
        """
        Linear Activation Function: f(x) = x

        Used in output layer for regression tasks.

        Args:
            Z (np.ndarray): Pre-activation values

        Returns:
            np.ndarray: Activated values (same as input)
        """
        return Z

    @staticmethod
    def linear_derivative(Z: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Derivative of Linear activation: df/dx = 1

        Args:
            Z (np.ndarray): Pre-activation values
            A (np.ndarray, optional): Activated values (not used for linear)

        Returns:
            np.ndarray: Derivative (ones)
        """
        return np.ones_like(Z)

    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        """
        ReLU (Rectified Linear Unit) Activation: f(x) = max(0, x)

        Widely used in hidden layers. Addresses vanishing gradient problem
        and enables faster training.

        Args:
            Z (np.ndarray): Pre-activation values

        Returns:
            np.ndarray: Activated values (0 if x < 0, x otherwise)
        """
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Derivative of ReLU activation: df/dx = 1 if x > 0, 0 if x <= 0

        Args:
            Z (np.ndarray): Pre-activation values
            A (np.ndarray, optional): Activated values (not used for ReLU)

        Returns:
            np.ndarray: Derivative (binary mask)
        """
        return (Z > 0).astype(float)

    @staticmethod
    def sigmoid(Z: np.ndarray) -> np.ndarray:
        """
        Sigmoid Activation: f(x) = 1 / (1 + e^(-x))

        Maps values to range (0, 1). Historically popular but prone to
        vanishing gradients in deep networks.

        Args:
            Z (np.ndarray): Pre-activation values

        Returns:
            np.ndarray: Activated values in range (0, 1)
        """
        # Clip to prevent overflow
        Z_clipped = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z_clipped))

    @staticmethod
    def sigmoid_derivative(Z: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Derivative of Sigmoid: df/dx = f(x) * (1 - f(x))

        Args:
            Z (np.ndarray): Pre-activation values (not used, included for consistency)
            A (np.ndarray): Activated values (sigmoid output)

        Returns:
            np.ndarray: Derivative
        """
        return A * (1 - A)

    @staticmethod
    def tanh(Z: np.ndarray) -> np.ndarray:
        """
        Tanh (Hyperbolic Tangent) Activation: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

        Maps values to range (-1, 1). Generally outperforms sigmoid in hidden layers.

        Args:
            Z (np.ndarray): Pre-activation values

        Returns:
            np.ndarray: Activated values in range (-1, 1)
        """
        return np.tanh(Z)

    @staticmethod
    def tanh_derivative(Z: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Derivative of Tanh: df/dx = 1 - f(x)^2

        Args:
            Z (np.ndarray): Pre-activation values (not used, included for consistency)
            A (np.ndarray): Activated values (tanh output)

        Returns:
            np.ndarray: Derivative
        """
        return 1 - (A ** 2)

    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        """
        Softmax Activation: f(x_i) = e^(x_i) / sum(e^(x_j))

        Converts outputs to probability distribution. Used in multi-class
        classification output layer.

        Args:
            Z (np.ndarray): Pre-activation values (shape: [batch_size, num_classes])

        Returns:
            np.ndarray: Probability distribution (sums to 1 across classes)
        """
        # Subtract max for numerical stability
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(Z: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Derivative of Softmax: df/dx_i = f(x_i) * (1 - f(x_i))

        Note: For multi-class problems with cross-entropy loss, this is often
        combined with the loss derivative for computational efficiency.

        Args:
            Z (np.ndarray): Pre-activation values (not used)
            A (np.ndarray): Activated values (softmax output)

        Returns:
            np.ndarray: Derivative
        """
        # Jacobian matrix for softmax
        return A * (1 - A)


# ==============================================================================
# PART 2: LOSS FUNCTIONS
# ==============================================================================

class LossFunctions:
    """
    Class containing loss (cost) functions for different tasks.
    """

    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error Loss: L = (1/m) * sum((y_pred - y_true)^2)

        Used for regression tasks.

        Args:
            y_true (np.ndarray): True labels/outputs (shape: [m, output_size])
            y_pred (np.ndarray): Predicted values (shape: [m, output_size])

        Returns:
            float: MSE loss value
        """
        m = y_true.shape[0]
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivative of MSE with respect to predictions: dL/dy_pred = 2/m * (y_pred - y_true)

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted values

        Returns:
            np.ndarray: Gradient of loss
        """
        m = y_true.shape[0]
        return 2 * (y_pred - y_true) / m

    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Cross-Entropy Loss: L = -1/m * sum(y_true * log(y_pred))

        Used for classification tasks. y_pred should be probabilities from softmax.

        Args:
            y_true (np.ndarray): One-hot encoded labels (shape: [m, num_classes])
            y_pred (np.ndarray): Predicted probabilities (shape: [m, num_classes])

        Returns:
            float: Cross-entropy loss value
        """
        m = y_true.shape[0]
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        return loss

    @staticmethod
    def cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivative of Cross-Entropy loss: dL/dy_pred = -(y_true / y_pred)

        When combined with softmax, simplifies to: dL/dZ = (y_pred - y_true)

        Args:
            y_true (np.ndarray): One-hot encoded labels
            y_pred (np.ndarray): Predicted probabilities

        Returns:
            np.ndarray: Gradient of loss
        """
        m = y_true.shape[0]
        return (y_pred - y_true) / m


# ==============================================================================
# PART 3: NEURAL NETWORK LAYER
# ==============================================================================

class Layer:
    """
    Represents a single layer in a neural network.

    Handles weights, biases, activations, and stores cache for backpropagation.
    """

    def __init__(self, input_size: int, output_size: int,
                 activation: str = 'relu', learning_rate: float = 0.01):
        """
        Initialize a neural network layer.

        Args:
            input_size (int): Number of input features
            output_size (int): Number of neurons in this layer
            activation (str): Activation function name ('linear', 'relu', 'sigmoid', 'tanh', 'softmax')
            learning_rate (float): Learning rate for weight updates
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate

        # Initialize weights with He initialization (good for ReLU)
        if activation == 'relu':
            self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)

        # Initialize biases to zero
        self.b = np.zeros((1, output_size))

        # For momentum optimization
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        # Cache for backward pass
        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the layer.

        Args:
            X (np.ndarray): Input data (shape: [batch_size, input_size])

        Returns:
            np.ndarray: Activated output (shape: [batch_size, output_size])
        """
        # Linear transformation: Z = X @ W + b
        Z = np.dot(X, self.W) + self.b

        # Apply activation function
        if self.activation == 'linear':
            A = ActivationFunctions.linear(Z)
        elif self.activation == 'relu':
            A = ActivationFunctions.relu(Z)
        elif self.activation == 'sigmoid':
            A = ActivationFunctions.sigmoid(Z)
        elif self.activation == 'tanh':
            A = ActivationFunctions.tanh(Z)
        elif self.activation == 'softmax':
            A = ActivationFunctions.softmax(Z)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        # Store for backward pass
        self.cache = {'X': X, 'Z': Z, 'A': A}

        return A

    def backward(self, dA: np.ndarray, momentum: float = 0.9) -> np.ndarray:
        """
        Backward propagation through the layer (backpropagation).

        Args:
            dA (np.ndarray): Gradient of loss with respect to activation (from next layer)
            momentum (float): Momentum coefficient for weight updates

        Returns:
            np.ndarray: Gradient with respect to input (dX) for previous layer
        """
        X = self.cache['X']
        Z = self.cache['Z']
        A = self.cache['A']
        m = X.shape[0]

        # Compute gradient with respect to pre-activation (dZ)
        if self.activation == 'linear':
            dZ = dA * ActivationFunctions.linear_derivative(Z, A)
        elif self.activation == 'relu':
            dZ = dA * ActivationFunctions.relu_derivative(Z, A)
        elif self.activation == 'sigmoid':
            dZ = dA * ActivationFunctions.sigmoid_derivative(Z, A)
        elif self.activation == 'tanh':
            dZ = dA * ActivationFunctions.tanh_derivative(Z, A)
        elif self.activation == 'softmax':
            # For softmax with cross-entropy, gradient is already (y_pred - y_true)
            dZ = dA
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

        # Compute gradients for weights and biases
        dW = np.dot(X.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m

        # Compute gradient with respect to input (for previous layer)
        dX = np.dot(dZ, self.W.T)

        # Update weights and biases using momentum
        self.vW = momentum * self.vW - self.learning_rate * dW
        self.vb = momentum * self.vb - self.learning_rate * db

        self.W += self.vW
        self.b += self.vb

        return dX


# ==============================================================================
# PART 4: DEEP NEURAL NETWORK
# ==============================================================================

class DeepNeuralNetwork:
    """
    Deep Neural Network with multiple layers.

    Implements forward propagation, loss computation, and backpropagation
    for training on regression or classification tasks.
    """

    def __init__(self, layer_sizes: List[int], activations: List[str],
                 loss_function: str = 'mse', learning_rate: float = 0.01):
        """
        Initialize the deep neural network.

        Args:
            layer_sizes (List[int]): Sizes of each layer (including input and output)
                                    e.g., [input_features, hidden1, hidden2, output]
            activations (List[str]): Activation function for each layer
                                    Length should be len(layer_sizes) - 1
            loss_function (str): 'mse' for regression, 'cross_entropy' for classification
            learning_rate (float): Learning rate for optimization
        """
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Number of activations must be layer_sizes - 1")

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.n_layers = len(layer_sizes) - 1

        # Initialize layers
        self.layers = []
        for i in range(self.n_layers):
            layer = Layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i],
                learning_rate=learning_rate
            )
            self.layers.append(layer)

        # Store training history
        self.train_losses = []
        self.val_losses = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through all layers.

        Args:
            X (np.ndarray): Input data (shape: [batch_size, input_features])

        Returns:
            np.ndarray: Network predictions (shape: [batch_size, output_size])
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss for the given predictions.

        Args:
            y_true (np.ndarray): True labels/outputs
            y_pred (np.ndarray): Predicted values

        Returns:
            float: Loss value
        """
        if self.loss_function == 'mse':
            return LossFunctions.mean_squared_error(y_true, y_pred)
        elif self.loss_function == 'cross_entropy':
            return LossFunctions.cross_entropy(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, momentum: float = 0.9):
        """
        Backward propagation through all layers (complete backpropagation).

        This is the core algorithm for training the network. It computes gradients
        and updates weights using gradient descent with momentum.

        Args:
            y_true (np.ndarray): True labels/outputs
            y_pred (np.ndarray): Predicted values from forward pass
            momentum (float): Momentum coefficient for optimization
        """
        # Compute initial gradient (dL/dA_output)
        if self.loss_function == 'mse':
            dA = LossFunctions.mse_derivative(y_true, y_pred)
        elif self.loss_function == 'cross_entropy':
            dA = LossFunctions.cross_entropy_derivative(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")

        # Backpropagate through layers in reverse order
        for i in range(self.n_layers - 1, -1, -1):
            dA = self.layers[i].backward(dA, momentum)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              momentum: float = 0.9, early_stopping_patience: int = 15):
        """
        Train the neural network using mini-batch gradient descent with backpropagation.

        Args:
            X_train (np.ndarray): Training input features
            y_train (np.ndarray): Training labels/outputs
            X_val (np.ndarray): Validation input features
            y_val (np.ndarray): Validation labels/outputs
            epochs (int): Number of training epochs
            batch_size (int): Mini-batch size
            momentum (float): Momentum coefficient
            early_stopping_patience (int): Patience for early stopping
        """
        n_samples = X_train.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                batch_loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                n_batches += 1

                # Backward pass (backpropagation)
                self.backward(y_batch, y_pred, momentum)

            # Average training loss for the epoch
            train_loss = epoch_loss / n_batches
            self.train_losses.append(train_loss)

            # Validation
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            self.val_losses.append(val_loss)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X (np.ndarray): Input features

        Returns:
            np.ndarray: Network predictions
        """
        return self.forward(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Evaluate the model on test data.

        Args:
            X_test (np.ndarray): Test input features
            y_test (np.ndarray): Test labels/outputs

        Returns:
            Tuple[float, np.ndarray]: (loss, predictions)
        """
        predictions = self.predict(X_test)
        loss = self.compute_loss(y_test, predictions)
        return loss, predictions

    def get_network_info(self) -> str:
        """
        Get information about the network architecture.

        Returns:
            str: Network information
        """
        info = "=" * 70 + "\n"
        info += "NETWORK ARCHITECTURE\n"
        info += "=" * 70 + "\n"
        info += f"Input Size: {self.layer_sizes[0]}\n"

        total_params = 0
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            params = layer.input_size * layer.output_size + layer.output_size
            total_params += params
            info += f"Layer {i+1}: {layer.input_size} → {layer.output_size} ({activation})\n"
            info += f"  Parameters: {params:,}\n"

        info += "-" * 70 + "\n"
        info += f"Total Parameters: {total_params:,}\n"
        info += f"Loss Function: {self.loss_function}\n"
        info += f"Learning Rate: {self.learning_rate}\n"
        info += "=" * 70 + "\n"

        return info


# ==============================================================================
# PART 5: DEMONSTRATION AND TESTING
# ==============================================================================

def create_regression_data(n_samples=1000, n_features=10, random_state=42):
    """Create synthetic regression dataset."""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.sum(X[:, :3], axis=1, keepdims=True) + np.random.randn(n_samples, 1) * 0.1
    y = y.astype(np.float32)

    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def create_classification_data(n_samples=1000, n_features=20, n_classes=3, random_state=42):
    """Create synthetic classification dataset."""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y_class = np.random.randint(0, n_classes, n_samples)

    # One-hot encode labels
    y = np.zeros((n_samples, n_classes))
    y[np.arange(n_samples), y_class] = 1

    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]


def main():
    """Main function demonstrating the neural network implementation."""

    print("\n" + "=" * 70)
    print("DEEP NEURAL NETWORK IMPLEMENTATION")
    print("Programming Assignment: Complete Implementation")
    print("=" * 70 + "\n")

    # ========================================================================
    # EXAMPLE 1: REGRESSION TASK
    # ========================================================================
    print("EXAMPLE 1: REGRESSION TASK")
    print("-" * 70)

    # Create data
    X_train, y_train, X_val, y_val = create_regression_data(n_samples=1000, n_features=10)

    # Define network architecture
    layer_sizes = [10, 64, 32, 16, 1]
    activations = ['relu', 'relu', 'relu', 'linear']

    # Create and train model
    print("Building regression model...")
    model_regression = DeepNeuralNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        loss_function='mse',
        learning_rate=0.01
    )

    print(model_regression.get_network_info())

    print("Training regression model...\n")
    model_regression.train(
        X_train, y_train, X_val, y_val,
        epochs=100,
        batch_size=32,
        momentum=0.9,
        early_stopping_patience=15
    )

    # Evaluate
    test_loss, test_pred = model_regression.evaluate(X_val, y_val)
    print(f"\nValidation MSE Loss: {test_loss:.6f}\n")

    # ========================================================================
    # EXAMPLE 2: CLASSIFICATION TASK
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE 2: MULTI-CLASS CLASSIFICATION TASK")
    print("-" * 70)

    # Create data
    X_train, y_train, X_val, y_val = create_classification_data(
        n_samples=1000, n_features=20, n_classes=3
    )

    # Define network architecture
    layer_sizes = [20, 64, 32, 3]
    activations = ['relu', 'relu', 'softmax']

    # Create and train model
    print("Building classification model...")
    model_classification = DeepNeuralNetwork(
        layer_sizes=layer_sizes,
        activations=activations,
        loss_function='cross_entropy',
        learning_rate=0.01
    )

    print(model_classification.get_network_info())

    print("Training classification model...\n")
    model_classification.train(
        X_train, y_train, X_val, y_val,
        epochs=100,
        batch_size=32,
        momentum=0.9,
        early_stopping_patience=15
    )

    # Evaluate
    test_loss, test_pred = model_classification.evaluate(X_val, y_val)
    print(f"\nValidation Cross-Entropy Loss: {test_loss:.6f}")

    # Classification accuracy
    pred_classes = np.argmax(test_pred, axis=1)
    true_classes = np.argmax(y_val, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    print(f"Validation Accuracy: {accuracy:.4f}\n")
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()