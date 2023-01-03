from ml_manager import MLManager
from Models.generic_ml_model import GenericMLModel
from Utils.data_container import DataContainer
from Utils.data_loader import DataLoader


class MLComparer:
    def __init__(self):
        self.data_container = DataContainer()

        self.all_models_training_specs = dict()
        self.all_models_benchmark_specs = dict()

    def initialize(self) -> None:
        self.data_container.load(DataLoader("./Data/Engineered/Asteroid_Updated_Engineered_Scaled.bin"))

    def compare_training(self, model_name: str, model: GenericMLModel, number_of_epochs: int) -> GenericMLModel:
        trained_model, training_specs = MLManager.train_model_get_specs(model, self.data_container, number_of_epochs)

        self.all_models_training_specs[model_name] = training_specs
        return trained_model

    def compare_benchmark(self, model_name: str, model: GenericMLModel) -> None:
        benchmark_specs = MLManager.benchmark_model(model, self.data_container)

        self.all_models_benchmark_specs[model_name] = benchmark_specs

    def dump_comparison(self) -> None:
        pass
