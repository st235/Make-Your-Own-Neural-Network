from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

class Module(ABC):
    """Abstract base class for a neural network building block.
    """

    def __init__(self):
        self.__forward_cache: Optional[np.ndarray] = None

    @property
    def _forward_cache(self) -> np.ndarray:
        """Returns cache from inference (aka forward pass). Can be `None`.
        """
        assert self.__forward_cache is not None, \
            "Forward cache cannot be null when accessed. Did you forget to run inference?"
        return self.__forward_cache

    @_forward_cache.setter
    def _forward_cache(self, cache: np.ndarray):
        """Sets forward cache and overrides previously saved value, if any.
        """
        self.__forward_cache = cache

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of a layer or block.

        A block takes an input with `in_features` and outputs a batch of vectors
        with `out_features`. In cases where `in_features` and `out_features` are
        the same, it's often clearer to refer to them as `num_features`.

        :param x: Input matrix of shape (batch_size, in_features)
        :return: Output matrix of shape (batch_size, out_features)
        """
        ...

    @abstractmethod
    def backward(self, grads_output: np.ndarray, lr: float) -> np.ndarray:
        """Backward pass of a layer or block.

        :param grads_output: Gradient propagated from the next layer
            (in the forward direction), with shape (batch_size, out_features)
        :param lr: Learning rate
        :return: Gradient propagated up to this block (including), with shape (batch_size, in_features)
        """
        ...
