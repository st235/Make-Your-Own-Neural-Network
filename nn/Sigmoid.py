import numpy as np
from typing import Optional

from nn.Module import Module

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.__input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.__input = x
        return Sigmoid.__sigmoid(x)

    def backward(self, grads_output: np.ndarray, lr: float)  -> np.ndarray:
        assert self.__input is not None
        return grads_output * Sigmoid.__sigmoid(self.__input) * (1 - Sigmoid.__sigmoid(self.__input))

    def __str__(self) -> str:
        return "Sigmoid"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def __sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
