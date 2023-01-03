import numpy as np
from data_loader import DataLoader


class DataContainer:
    def __init__(self, training_set: tuple = None, validation_set: tuple = None, test_set: tuple = None):
        self.training_set = training_set
        self.validation_set = validation_set
        self.test_set = test_set

    def load(self, data_loader: DataLoader) -> None:
        feature_set, target_set = data_loader.load()

        assert feature_set.shape[1] == target_set.shape[0]
        permutation = np.random.permutation(feature_set.shape[1])

        shuffled_feature_set, shuffled_target_set = feature_set[permutation, :], target_set[permutation]

        training_set_upper_edge = 0.8 * feature_set.shape[1]
        validation_set_upper_edge = 0.9 * feature_set.shape[1]

        self.training_set = shuffled_feature_set[:training_set_upper_edge, :], shuffled_target_set[:training_set_upper_edge]
        self.validation_set = shuffled_feature_set[training_set_upper_edge:validation_set_upper_edge, :], shuffled_target_set[training_set_upper_edge:validation_set_upper_edge]
        self.test_set = shuffled_feature_set[validation_set_upper_edge:, :], shuffled_target_set[validation_set_upper_edge:]

    def get_training_set(self) -> tuple:
        return self.training_set

    def get_validation_set(self) -> tuple:
        return self.validation_set

    def get_test_set(self) -> tuple:
        return self.test_set

    def get_input_shape(self) -> int:
        return self.training_set[0].shape[1]
