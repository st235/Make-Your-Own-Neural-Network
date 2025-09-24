import math

import numpy as np

from nn.WeightsInitialiser import WeightsInitialiser


class NormalisedXavierWeightsInitialiser(WeightsInitialiser):
    def __init__(self):
        super().__init__()

    def initialise(self, in_features: float, out_features: float) -> np.ndarray:
        return np.random.uniform(
            low=-math.sqrt(6 / (in_features + out_features)),
            high=math.sqrt(6 / (in_features + out_features)),
            size=(in_features, out_features),
        )
