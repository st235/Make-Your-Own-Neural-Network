from abc import ABC, abstractmethod

import numpy as np


class WeightsInitialiser(ABC):
    """Random weights initialisation heuristic for nn.Weights block.
    """

    def __init__(self):
        pass

    @abstractmethod
    def initialise(self, in_features: float, out_features: float) -> np.ndarray:
        ...
