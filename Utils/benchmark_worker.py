import numpy as np
from sklearn.metrics import r2_score
from Models.generic_ml_model import GenericMLModel


class BenchmarkWorker:
    def __init__(self):
        pass

    @staticmethod
    def mean_squared_error(model: GenericMLModel, data: tuple) -> float:
        features, target = data

        prediction = model.predict_on_array(features)
        delta = prediction - target
        return float(np.sum(np.square(delta))) / features.shape[0]

    @staticmethod
    def r2_score_value(model: GenericMLModel, data: tuple) -> tuple:
        features, target = data
        return r2_score(target, model.predict_on_array(features))
