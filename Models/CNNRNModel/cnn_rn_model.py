import keras.models
import numpy as np
from Models.generic_ml_model import GenericMLModel
from keras import backend
from keras.layers import Dense, Input, Dropout, Conv1D, Flatten
from keras.models import Model
from keras.optimizers import Adam


class CNNRNModel(GenericMLModel):
    def __init__(self, rn_model: Model = None):
        super().__init__()
        self.rn_model = rn_model
        self.learning_rate = 0.00005

    def init_for_training(self, input_shape: int) -> None:
        input_layer = Input(shape=(input_shape, 1))

        cov_layer_1 = Conv1D(filters=64, kernel_size=2, activation='relu')(input_layer)
        flatten_layer = Flatten()(cov_layer_1)

        dense_layer_1_reg = Dense(512, kernel_initializer='lecun_normal', activation="selu")(flatten_layer)
        dense_layer_2_reg = Dense(256, kernel_initializer='lecun_normal', activation="selu")(dense_layer_1_reg)

        output_layer = Dense(1, activation="relu")(dense_layer_2_reg)

        self.rn_model = Model(input_layer, output_layer)
        self.rn_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")

    def predict(self, features: np.array) -> float:
        return self.rn_model.predict(features)[0]

    def predict_on_array(self, features: np.array) -> np.array:
        return self.rn_model.predict(features).reshape(-1)

    def train(self, data: tuple[np.array, np.array]) -> None:
        self.rn_model.fit(data[0], data[1])

    def save(self, path_to_model: str) -> None:
        self.rn_model.save(path_to_model)

    def load(self, path_to_model: str) -> None:
        self.rn_model = keras.models.load_model(path_to_model)

    def copy(self) -> any:
        new_obj = CNNRNModel()
        new_obj.learning_rate = self.learning_rate
        new_obj.rn_model = keras.models.clone_model(self.rn_model)
        new_obj.rn_model.set_weights(self.rn_model.get_weights())
        return new_obj
