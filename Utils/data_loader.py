import pickle
import numpy as np


class DataLoader:
    def __init__(self, binary_file_data_path: str):
        self.binary_file_data_path = binary_file_data_path

    def load(self) -> tuple[np.array, np.array]:
        with open(self.binary_file_data_path, "rb") as file_descriptor:
            features, targets = pickle.load(file_descriptor)

        return features, targets

