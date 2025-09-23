import numpy as np
from typing import Optional

from nn.Module import Module

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.__input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.__input = x
        return np.maximum(0, x)

    def backward(self, grads_output: np.ndarray, lr: float)  -> np.ndarray:
        assert self.__input is not None
        return np.multiply(grads_output, 1.0 * (self.__input > 0))

    def __str__(self) -> str:
        return "ReLU"

    def __repr__(self) -> str:
        return str(self)
