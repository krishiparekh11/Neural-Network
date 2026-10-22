import numpy as np


class Softmax:

    def __init__(self):
        self.output = None

    def forward(self, X):
        
        # Handle both 1D and 2D inputs
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Numerical stability: subtract max from each row
        # This prevents overflow when computing exp()
        X_shifted = X - np.max(X, axis=1, keepdims=True)

        # Compute exponentials
        exp_X = np.exp(X_shifted)

        # Normalize by sum to get probabilities
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)

        return self.output

    def backward(self, dL_dout):
        batch_size = self.output.shape[0]
        dL_dinput = np.zeros_like(self.output)

        for i in range(batch_size):
            s = self.output[i].reshape(-1, 1)  # Column vector
            jacobian = np.diagflat(s) - s @ s.T
            dL_dinput[i] = jacobian @ dL_dout[i]

        return dL_dinput

    def __call__(self, X):
        """Allow the activation to be called as a function."""
        return self.forward(X)


# Example usage and testing
if __name__ == "__main__":
    # Create an instance of Softmax
    softmax = Softmax()

    # Test 1: Single sample (1D input)
    print("Test 1: Single sample")
    x1 = np.array([1.0, 2.0, 3.0])
    output1 = softmax(x1)
    print(f"Input: {x1}")
    print(f"Output: {output1}")
    print(f"Sum (should be 1.0): {np.sum(output1)}\n")

    # Test 2: Batch of samples (2D input)
    print("Test 2: Batch of samples")
    x2 = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [10.0, 20.0, 30.0]  # Large values to test numerical stability
    ])
    output2 = softmax(x2)
    print(f"Input shape: {x2.shape}")
    print(f"Output:\n{output2}")
    print(f"Row sums (should all be 1.0): {np.sum(output2, axis=1)}\n")

    # Test 3: Numerical stability with large values
    print("Test 3: Numerical stability check")
    x3 = np.array([1000.0, 1001.0, 1002.0])
    output3 = softmax(x3)
    print(f"Input: {x3}")
    print(f"Output: {output3}")
    print(f"No inf/nan values: {not (np.any(np.isinf(output3)) or np.any(np.isnan(output3)))}\n")

    # Test 4: Backward pass
    print("Test 4: Backward pass")
    x4 = np.array([[1.0, 2.0, 3.0]])
    output4 = softmax(x4)
    dL_dout = np.array([[0.1, 0.2, 0.3]])
    dL_dinput = softmax.backward(dL_dout)
    print(f"Forward output: {output4}")
    print(f"Gradient w.r.t. input: {dL_dinput}")