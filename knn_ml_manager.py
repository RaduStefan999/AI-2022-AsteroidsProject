from ml_manager import MLManager
from Models.generic_ml_model import GenericMLModel
from Utils.data_container import DataContainer


class KNNManager:
    def __init__(self):
        pass

    @staticmethod
    def train_model_get_specs(model: GenericMLModel, data_container: DataContainer, max_k: int) \
            -> tuple[GenericMLModel, dict[str, list]]:

        mean_squared_error_best = 999999999999
        best_model = None

        all_training_specs = dict()

        for k in range(1, max_k, 2):
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
