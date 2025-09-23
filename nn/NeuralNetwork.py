import numpy as np

from typing import List, Callable

from nn.Module import Module
from nn.Weights import Weights
from nn.Sigmoid import Sigmoid
from nn.Softmax import Softmax
from nn.ReLU import ReLU

class NeuralNetwork:
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_layers: tuple[int] = (),
                 hidden_layer_activation: Callable[[], Module] = ReLU):
        self.__layers: List[Module] = []

        current_layer = in_features
        for hidden_layer in hidden_layers:
            self.__layers.append(Weights(in_features=current_layer, out_features=hidden_layer))
            self.__layers.append(hidden_layer_activation())
            current_layer = hidden_layer

        self.__layers.append(Weights(in_features=current_layer, out_features=out_features))
        self.__layers.append(Softmax())

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x
        for i, layer in enumerate(self.__layers):
            y = layer.forward(y)
        return y

    def backward(self,
                 grads: np.ndarray,
                 lr: float):
        for layer in reversed(self.__layers):
            grads = layer.backward(grads, lr)

    def __str__(self) -> str:
        layers_description = "\n".join([f"\t{str(layer)}" for layer in self.__layers])
        return f"[\n{layers_description}\n]"

    def __repr__(self) -> str:
        return str(self)




