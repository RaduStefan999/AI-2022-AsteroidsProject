import joblib
import copy
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from Models.generic_ml_model import GenericMLModel


class SVMModel(GenericMLModel):
    def __init__(self, regressor: SVR = None):
        super().__init__()
        self.regressor = regressor

    def init_for_training(self, input_shape: int) -> None:
        self.regressor = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

    def predict(self, features: np.array) -> float:
        return self.regressor.predict(features)[0]

    def predict_on_array(self, features: np.array) -> np.array:
        return self.regressor.predict(features).reshape(-1)

    def train(self, data: tuple[np.array, np.array]) -> None:
        self.regressor.fit(data[0], data[1])

    def save(self, path_to_model: str) -> None:
        joblib.dump(self.regressor, path_to_model)

    def load(self, path_to_model: str) -> None:
        self.regressor = joblib.load(path_to_model)

    def copy(self) -> any:
        return copy.deepcopy(self)
