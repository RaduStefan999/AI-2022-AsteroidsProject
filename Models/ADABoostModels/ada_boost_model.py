import copy
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from Models.generic_ml_model import GenericMLModel


class ADABoostModel(GenericMLModel):
    def __init__(self, regressor: AdaBoostRegressor = None):
        super().__init__()
        self.regressor = regressor

    def init_for_training(self, input_shape: int) -> None:
        self.regressor = AdaBoostRegressor(n_estimators=100, random_state=0)

    def predict(self, features: np.array) -> float:
        return self.regressor.predict(features)[0]

    def predict_on_array(self, features: np.array) -> np.array:
        return self.regressor.predict(features).reshape(-1)

    def train(self, data: tuple[np.array, np.array]) -> None:
        self.regressor.fit(data[0], data[1])

    def save(self, path_to_model: str) -> None:
        raise NotImplementedError()

    def load(self, path_to_model: str) -> None:
        raise NotImplementedError()

    def copy(self) -> any:
        return copy.deepcopy(self)
