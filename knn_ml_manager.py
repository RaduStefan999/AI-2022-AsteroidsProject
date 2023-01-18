import numpy as np

from ml_manager import MLManager
from Utils.benchmark_worker import BenchmarkWorker
from Models.generic_ml_model import GenericMLModel
from Utils.data_container import DataContainer

class KNNManager(MLManager):
    def __init__(self):
        pass

    @staticmethod
    def train_model_get_specs(model: GenericMLModel, data_container: DataContainer, max_k: int) \
            -> tuple[GenericMLModel, dict[str, list]]:

        mean_squared_error_best = 999999999999
        best_model = None

        all_training_specs = dict()

        for k in range(1, max_k):
            current_model = model.copy()
            current_model.neighbours = k
            current_model.init_for_training(data_container.get_input_shape())

            current_model.train(data_container.get_shuffled_training_set())

            current_specs = MLManager.get_specs(current_model, data_container.get_validation_set())
            all_training_specs = MLManager.concat_specs(all_training_specs, current_specs)

            print("\n".join([f"KNN for k = {k} {key}: {current_specs[key]}" for key in current_specs.keys()]))

            if current_specs["mean_squared_error"] < mean_squared_error_best:
                mean_squared_error_best = current_specs["mean_squared_error"]
                # TODO: remove redundant copy operation?
                best_model = current_model.copy()

        return best_model, all_training_specs

    @staticmethod
    def train_model(model: GenericMLModel, data_container: DataContainer, nr_of_epochs: int) -> GenericMLModel:
        return MLManager.train_model_get_specs(model, data_container, nr_of_epochs)[0]

    @staticmethod
    def benchmark_model(model: GenericMLModel, data_container: DataContainer) -> dict[str, float]:
        validation_specs = MLManager.get_specs(model, data_container.get_validation_set())
        print("\n".join([f"Last validation set {key}: {validation_specs[key]}" for key in validation_specs.keys()]))

        test_specs = MLManager.get_specs(model, data_container.get_test_set())
        print("\n".join([f"Test set {key}: {test_specs[key]}" for key in test_specs.keys()]))

        return validation_specs

    @staticmethod
    def concat_specs(all_training_specs: dict, current_specs: dict[str, float]) -> dict[str, list]:
        return {key: all_training_specs.get(key, []) + [current_specs[key]] for key in current_specs.keys()}

    @staticmethod
    def get_specs(model: GenericMLModel, data_set: tuple) -> dict[str, float]:
        features, target = data_set
        prediction = model.predict_on_array(features)

        return {
            "mean_absolute_error": BenchmarkWorker.mean_absolute_error(data_set, prediction),
            "mean_squared_error": BenchmarkWorker.mean_squared_error(data_set, prediction),
            "median_absolute_error": BenchmarkWorker.median_absolute_error(data_set, prediction),
            "explained_variance_score": BenchmarkWorker.explained_variance_score_value(data_set, prediction),
            "r2_score": BenchmarkWorker.r2_score_value(data_set, prediction)
        }
