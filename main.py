from Utils.data_container import DataContainer
from Utils.data_loader import DataLoader
from Models.RNModels.rn_model import RNModel
from ml_manager import MLManager

if __name__ == '__main__':
    model = RNModel()
    data_container = DataContainer()
    data_container.load(DataLoader("./Data/Engineered/Asteroid_Updated_Engineered.bin"))

    model = MLManager.train_model(model, data_container)
    MLManager.benchmark_model(model, data_container)
