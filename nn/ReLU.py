import numpy as np

from nn.Module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._forward_cache = x
        return np.maximum(0, x)

    def backward(self, grads_output: np.ndarray, lr: float)  -> np.ndarray:
        return np.multiply(grads_output, 1.0 * (self._forward_cache > 0))

    def __str__(self) -> str:
        return "ReLU"

    def __repr__(self) -> str:
        return str(self)
