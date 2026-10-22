import numpy as np
import os
from pathlib import Path

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting will be skipped.")


class DeepNeuralNetwork:
    """
    A deep neural network with multiple hidden layers using NumPy.

    Architecture:
    - Input layer
    - Multiple hidden layers with ReLU activation
    - Output layer
    - Batch normalization and dropout for regularization
    """

    def __init__(self, layer_sizes, learning_rate=0.001, dropout_rate=0.3):
        """
        Initialize the deep neural network.

        Args:
            layer_sizes (list): List of layer sizes, e.g., [20, 128, 64, 32, 16, 8, 1]
                               First element is input size, last is output size
            learning_rate (float): Learning rate for gradient descent
            dropout_rate (float): Dropout probability for regularization
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.n_layers = len(layer_sizes) - 1

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.velocities_w = []  # For momentum
        self.velocities_b = []

        for i in range(self.n_layers):
            # He initialization for better convergence
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))

            self.weights.append(w)
            self.biases.append(b)
            self.velocities_w.append(np.zeros_like(w))
            self.velocities_b.append(np.zeros_like(b))

        self.train_losses = []
        self.val_losses = []

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)

    def forward(self, X, training=True):
        """
        Forward pass through the network.

        Args:
            X (np.ndarray): Input data
            training (bool): Whether in training mode (for dropout)

        Returns:
            np.ndarray: Output predictions
            dict: Cache for backward pass
        """
        cache = {'A': [X], 'Z': [], 'D': []}  # D for dropout masks
        A = X

        for i in range(self.n_layers):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            cache['Z'].append(Z)

            # Apply ReLU to hidden layers, no activation to output layer
            if i < self.n_layers - 1:
                A = self.relu(Z)

                # Dropout during training
                if training:
                    D = np.random.binomial(1, 1 - self.dropout_rate, A.shape) / (1 - self.dropout_rate)
                    A = A * D
                    cache['D'].append(D)
                else:
                    cache['D'].append(None)
            else:
                A = Z  # Output layer (no activation)
                cache['D'].append(None)

            cache['A'].append(A)

        return A, cache

    def backward(self, X, y, predictions, cache):
        """
        Backward pass for computing gradients.

        Args:
            X (np.ndarray): Input data
            y (np.ndarray): Target data
            predictions (np.ndarray): Network predictions
            cache (dict): Cache from forward pass

        Returns:
            list, list: Gradients for weights and biases
        """
        m = X.shape[0]
        grad_w = []
        grad_b = []

        # Compute output layer gradient (MSE loss)
        dA = (predictions - y) / m

        # Backprop through layers
        for i in range(self.n_layers - 1, -1, -1):
            dZ = dA @ self.weights[i].T

            if i > 0:
                dZ = dZ * self.relu_derivative(cache['Z'][i-1])
                if cache['D'][i-1] is not None:
                    dZ = dZ * cache['D'][i-1]

            # Compute weight and bias gradients
            dW = cache['A'][i].T @ dA
            dB = np.sum(dA, axis=0, keepdims=True)

            grad_w.insert(0, dW)
            grad_b.insert(0, dB)

            dA = dZ

        return grad_w, grad_b

    def update_weights(self, grad_w, grad_b, momentum=0.9):
        """
        Update weights using momentum-based gradient descent.

        Args:
            grad_w (list): Weight gradients
            grad_b (list): Bias gradients
            momentum (float): Momentum coefficient
        """
        for i in range(self.n_layers):
            # Update velocities (momentum)
            self.velocities_w[i] = momentum * self.velocities_w[i] - self.learning_rate * grad_w[i]
            self.velocities_b[i] = momentum * self.velocities_b[i] - self.learning_rate * grad_b[i]

            # Update parameters
            self.weights[i] += self.velocities_w[i]
            self.biases[i] += self.velocities_b[i]

    def train_epoch(self, X_train, y_train, batch_size=32):
        """
        Train for one epoch with mini-batch gradient descent.

        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training targets
            batch_size (int): Batch size for training

        Returns:
            float: Average training loss
        """
        n_samples = X_train.shape[0]
        indices = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Forward pass
            predictions, cache = self.forward(X_batch, training=True)

            # Compute loss (MSE)
            loss = np.mean((predictions - y_batch) ** 2)
            total_loss += loss
            n_batches += 1

            # Backward pass
            grad_w, grad_b = self.backward(X_batch, y_batch, predictions, cache)

            # Update weights
            self.update_weights(grad_w, grad_b)

        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, X_val, y_val):
        """
        Validate the model.

        Args:
            X_val (np.ndarray): Validation data
            y_val (np.ndarray): Validation targets

        Returns:
            float: Average validation loss
        """
        predictions, _ = self.forward(X_val, training=False)
        loss = np.mean((predictions - y_val) ** 2)
        self.val_losses.append(loss)
        return loss

    def train(self, X_train, y_train, X_val, y_val, epochs=100, early_stopping_patience=15):
        """
        Train the model with early stopping.

        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation data
            y_val (np.ndarray): Validation targets
            epochs (int): Number of training epochs
            early_stopping_patience (int): Patience for early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(X_train, y_train, batch_size=32)
            val_loss = self.validate(X_val, y_val)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Predictions
        """
        predictions, _ = self.forward(X, training=False)
        return predictions

    def plot_losses(self, save_path=None):
        """
        Plot training and validation losses.

        Args:
            save_path (str, optional): Path to save the plot
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping plot generation.")
            return

        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, label='Training Loss', linewidth=2, marker='o', markersize=3)
            plt.plot(self.val_losses, label='Validation Loss', linewidth=2, marker='s', markersize=3)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss (MSE)', fontsize=12)
            plt.title('Deep Neural Network Training Progress', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                # Create directory if it doesn't exist
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"✓ Plot saved to: {save_path}")
            else:
                plt.show()

            plt.close()
        except Exception as e:
            print(f"Error saving plot: {e}")


def create_sample_data(n_samples=1000, n_features=20, random_state=42):
    """
    Create synthetic data for testing.

    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        random_state (int): Random seed

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    np.random.seed(random_state)

    # Generate random input features
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Generate target (linear combination + noise)
    y = np.sum(X[:, :5], axis=1, keepdims=True) + np.random.randn(n_samples, 1) * 0.1
    y = y.astype(np.float32)

    # Split into train and validation
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    return X_train, y_train, X_val, y_val


def main():
    """Main function to demonstrate the deep neural network."""

    print("=" * 70)
    print("Deep Neural Network Implementation (NumPy-based)")
    print("=" * 70)
    print()

    # Create sample data
    print("Step 1: Creating synthetic dataset...")
    X_train, y_train, X_val, y_val = create_sample_data(n_samples=1000, n_features=20)
    print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print()

    # Define network architecture
    input_size = 20
    hidden_sizes = [128, 64, 32, 16, 8]  # 5 hidden layers
    output_size = 1

    layer_sizes = [input_size] + hidden_sizes + [output_size]

    print("Step 2: Building deep neural network...")
    print(f"  Architecture: {' → '.join(map(str, layer_sizes))}")
    print(f"  Total hidden layers: {len(hidden_sizes)}")
    print()

    # Initialize model
    model = DeepNeuralNetwork(layer_sizes, learning_rate=0.001, dropout_rate=0.3)

    # Count parameters
    total_params = sum(w.size + b.size for w, b in zip(model.weights, model.biases))
    print(f"  Total parameters: {total_params:,}")
    print()

    # Train the model
    print("Step 3: Training the network...")
    print("-" * 70)
    model.train(X_train, y_train, X_val, y_val, epochs=100, early_stopping_patience=15)
    print("-" * 70)
    print()

    # Plot training progress
    print("Step 4: Generating training plot...")

    # Determine output path based on OS
    home_dir = str(Path.home())
    plot_path = os.path.join(home_dir, "training_loss.png")

    model.plot_losses(save_path=plot_path)
    print()

    # Make predictions
    print("Step 5: Evaluating on validation set...")
    predictions = model.predict(X_val)

    # Calculate metrics
    mse = np.mean((predictions - y_val) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_val))

    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print()

    print("=" * 70)
    print("✓ Deep Neural Network training completed successfully!")
    print("=" * 70)

    return model, X_val, y_val, predictions


if __name__ == "__main__":
    model, X_val, y_val, predictions = main()