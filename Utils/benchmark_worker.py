import numpy as np
from sklearn.metrics import r2_score, explained_variance_score
from Models.generic_ml_model import GenericMLModel


class BenchmarkWorker:
    def __init__(self):
        pass

    @staticmethod
    def mean_absolute_error(data_set: tuple, prediction: np.array) -> float:
        features, target = data_set

        delta = target - prediction
        return float(np.sum(np.absolute(delta)) / features.shape[0])

    @staticmethod
    def mean_squared_error(data_set: tuple, prediction: np.array) -> float:
        features, target = data_set

        delta = target - prediction
        return float(np.sum(np.square(delta)) / features.shape[0])

    @staticmethod
    def median_absolute_error(data_set: tuple, prediction: np.array) -> float:
        features, target = data_set

        delta = target - prediction
        return float(np.median(np.absolute(delta)))

    @staticmethod
    def explained_variance_score_value(data_set: tuple, prediction: np.array) -> float:
        features, target = data_set
        return float(explained_variance_score(target, prediction))

    @staticmethod
    def r2_score_value(data_set: tuple, prediction: np.array) -> float:
        features, target = data_set
        return float(r2_score(target, prediction))
