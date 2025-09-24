import numpy as np

from nn.WeightsInitialiser import WeightsInitialiser


class HeWeightsInitialiser(WeightsInitialiser):
    def __init__(self):
        super().__init__()

    def initialise(self, in_features: float, out_features: float) -> np.ndarray:
        return np.random.normal(
            loc=0.0,
            scale=pow(2.0 / in_features, 0.5),
            size=(in_features, out_features),
        )
