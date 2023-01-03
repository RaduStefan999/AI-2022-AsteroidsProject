import numpy as np


class GenericModel:
    def __init__(self):
        pass

    def predict(self, features: np.array) -> float:
        raise NotImplementedError()
