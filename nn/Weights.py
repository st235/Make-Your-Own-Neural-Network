import numpy as np

from nn.Module import Module

class Weights(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()

        self.__in_features = in_features
        self.__out_features = out_features

        # Randomly initialised weights matrix of size in_features x out_features.
        # Uses He initialisation by default.
        self.__weights = np.random.normal(
            loc=0.0,
            scale=pow(2.0 / (self.__in_features + self.__out_features), 0.5),
            size=(self.__in_features, self.__out_features)
        )

        self.__bias = np.zeros((1, self.__out_features))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of a weights layer: z = xW + b.

        :param x: (batch_size, in_features)
        :return: (batch_size, out_features)
        """
        self._forward_cache = x
        return np.dot(x, self.__weights) + self.__bias

    def backward(self, grads_output: np.ndarray, lr: float) -> np.ndarray:
        """Backward pass of a weights layer.

        :param grads_output: (batch_size, out_features)
        :param lr: float
        :return: gradients prior applying this layer by size of (batch_size, in_features)
        """
        batch_size = self._forward_cache.shape[0]

        # __input is (batch_size, in_features), grads_output is (batch_size, out_features).
        grads_weight = np.dot(self._forward_cache.T, grads_output) / batch_size
        grads_bias = np.sum(grads_output, axis=0, keepdims=True) / batch_size
        grads_input = np.dot(grads_output, self.__weights.T)

        # Updating weights.
        self.__weights -= lr * grads_weight
        self.__bias -= lr * grads_bias

        return grads_input

    def __str__(self) -> str:
        return f"Weights({self.__in_features}, {self.__out_features})"

    def __repr__(self) -> str:
        return str(self)
