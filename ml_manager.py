from Models.generic_ml_model import GenericMLModel
from Utils.data_container import DataContainer


class MLManager:
    def __init__(self):
        pass

    @staticmethod
    def train_model(model: GenericMLModel, data_container: DataContainer) -> GenericMLModel:
        model.init_for_training(data_container.get_input_shape())
        for it in range(0, 200):
            model.train(data_container.get_shuffled_training_set())
        return model

    @staticmethod
    def load_model(model: GenericMLModel, path_to_model_data: str) -> GenericMLModel:
        pass

    @staticmethod
    def benchmark_model(model: GenericMLModel, data_container: DataContainer) -> None:
        pass
