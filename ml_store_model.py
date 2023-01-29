import shutil
import os
from Models.generic_ml_model import GenericMLModel


class MLStoreModel:
    def __init__(self):
        pass

    @staticmethod
    def make_path(model_name: str) -> None:
        model_path = os.path.join("./StoredModels", model_name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        os.mkdir(model_path)

    @staticmethod
    def store(model_name: str, model: GenericMLModel) -> None:
        MLStoreModel.make_path(model_name)
        model_path = os.path.join("./StoredModels", model_name)
        model.save(model_path)

    @staticmethod
    def load(model_name: str, model: GenericMLModel) -> GenericMLModel:
        model_path = os.path.join("./StoredModels", model_name)
        assert os.path.exists(model_path)

        model.load(model_path)
        return model
