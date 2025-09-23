import numpy as np

from nn.Module import Module

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        sigmoid = Sigmoid.__sigmoid(x)
        self._forward_cache = sigmoid
        return sigmoid

    def backward(self, grads_output: np.ndarray, lr: float)  -> np.ndarray:
        return grads_output * self._forward_cache * (1 - self._forward_cache)

    def __str__(self) -> str:
        return "Sigmoid"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def __sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
