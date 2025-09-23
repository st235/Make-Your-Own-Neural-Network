import numpy as np

from nn.Module import Module

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        softmax = Softmax.__softmax(x)
        self._forward_cache = softmax
        return softmax

    def backward(self, grads_output: np.ndarray, lr: float)  -> np.ndarray:
        batch_size = self._forward_cache.shape[0]
        grads_input = np.zeros_like(grads_output)

        softmax = self._forward_cache
        for i in range(batch_size):
            svec = softmax[i, :].T
            smat = np.tile(svec, softmax.shape[0])
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
