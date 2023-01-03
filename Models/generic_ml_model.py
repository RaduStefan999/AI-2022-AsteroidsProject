import numpy as np
from generic_model import GenericModel


class GenericMLModel(GenericModel):
    def __init__(self):
        super().__init__()

    def init_for_training(self, input_shape: int) -> None:
        raise NotImplementedError()

    def predict(self, features: np.array) -> float:
        raise NotImplementedError()

    def train(self, data: tuple[np.array, np.array]) -> None:
        raise NotImplementedError()

    def save(self, path_to_model: str) -> None:
        raise NotImplementedError()

    def load(self, path_to_model: str) -> None:
        raise NotImplementedError()

