from Utils.benchmark_worker import BenchmarkWorker
from Models.generic_ml_model import GenericMLModel
from Utils.data_container import DataContainer


class MLManager:
    def __init__(self):
        pass

    @staticmethod
    def train_model(model: GenericMLModel, data_container: DataContainer, nr_of_epochs: int) -> GenericMLModel:
        model.init_for_training(data_container.get_input_shape())
        for it in range(0, nr_of_epochs):
            model.train(data_container.get_shuffled_training_set())

            print(f"Iteration: {it} with mean_squared_error: "
                  f"{BenchmarkWorker.mean_squared_error(model, data_container.get_validation_set())}")

            print(f"Iteration: {it} with r2_score: "
                  f"{BenchmarkWorker.r2_score_value(model, data_container.get_validation_set())}")

        return model

    @staticmethod
    def load_model(model: GenericMLModel, path_to_model_data: str) -> GenericMLModel:
        pass

    @staticmethod
    def benchmark_model(model: GenericMLModel, data_container: DataContainer) -> None:
        print(f"Test set mean_squared_error: "
              f"{BenchmarkWorker.mean_squared_error(model, data_container.get_test_set())}")
