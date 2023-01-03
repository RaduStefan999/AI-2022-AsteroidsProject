from Utils.data_container import DataContainer
from Utils.data_loader import DataLoader
from Models.RNModels.rn_model import RNModel
from Models.ADABoostModels.ada_boost_model import ADABoostModel
from ml_manager import MLManager


def benchmark_model(mode_data: tuple) -> None:
    current_model, number_of_epochs = mode_data

    data_container = DataContainer()
    data_container.load(DataLoader("./Data/Engineered/Asteroid_Updated_Engineered_Scaled.bin"))

    current_model = MLManager.train_model(current_model, data_container, number_of_epochs)
    MLManager.benchmark_model(current_model, data_container)


def benchmark_models(models_data: list) -> None:
    for model_data in models_data:
        benchmark_model(model_data)


if __name__ == '__main__':
    benchmark_models([(RNModel(), 10)])
    #benchmark_models([(ADABoostModel(), 10)])

