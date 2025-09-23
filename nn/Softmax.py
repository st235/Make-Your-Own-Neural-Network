import numpy as np
from typing import Optional

from nn.Module import Module

class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.__input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.__input = x
        return Softmax.__softmax(x)

    def backward(self, grads_output: np.ndarray, lr: float)  -> np.ndarray:
        assert self.__input is not None

        batch_size = self.__input.shape[0]
        grads_input = np.zeros_like(grads_output)

        s = Softmax.__softmax(self.__input)

        for i in range(batch_size):
            svec = s[i, :].T
            smat = np.tile(svec, s.shape[0])
            grads_softmax_i = np.diag(svec.T) - np.dot(smat, smat.T)
            grads_input[i] = np.dot(grads_output[i], grads_softmax_i)

        return grads_input

    def __str__(self) -> str:
        return "Softmax"

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def __softmax(x: np.ndarray):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return np.divide(exp, np.sum(exp, axis=1, keepdims=True))
