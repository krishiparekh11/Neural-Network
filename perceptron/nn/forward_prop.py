"""Forward propagation: X -> Z, A for every layer."""


class ForwardProp:
    def __init__(self, model):
        self.model = model

    def run(self, X):
        cache = []
        A = X
        for layer in self.model.layers:
            Z, A_next = layer.forward(A)
            cache.append((A, Z))          # input to this layer and its Z
            A = A_next
        return A, cache
