import numpy as np

from abc import ABC, abstractmethod

class Module(ABC):
    """Abstract building block of a neural network.
    """

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of a layer or a block.
        A block accepts in_features and produces out_features batches of vectors.
        Sometimes, in_features and out_features are the same. In that case,
        it would be better to call it num_features.

        :param x: block input of size (batch_size, in_features)
        :return: block output of size (batch_size, out_features)
        """
        ...

    @abstractmethod
    def backward(self, grads_output: np.ndarray, lr: float) -> np.ndarray:
        """Backward pass of a layer or a block.

        :param grads_output: cumulative gradient up to the next layer
        (looking in a forward direction) of size (batch_size, out_features)
        :param lr: learning rate
        :return: cumulative gradient up to this block including of size (batch_size, out_features)
        """
        ...
