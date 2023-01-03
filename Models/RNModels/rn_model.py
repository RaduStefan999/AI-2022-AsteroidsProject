import keras.models
import numpy as np
from Models.generic_ml_model import GenericMLModel
from keras import backend
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam


class RNModel(GenericMLModel):
    def __init__(self, rn_model: Model = None):
        super().__init__()
        self.rn_model = rn_model
        self.learning_rate = 0.0001

    def init_for_training(self, input_shape: int) -> None:
        input_layer = Input(shape=input_shape)

        dense_layer_1 = Dense(256, kernel_initializer='lecun_normal', activation="selu")(input_layer)
        dense_layer_2 = Dense(128, kernel_initializer='lecun_normal', activation="selu")(dense_layer_1)

        output_layer = Dense(1, activation="linear")(dense_layer_2)

        self.rn_model = Model(input_layer, output_layer)
        self.rn_model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")

    def predict(self, features: np.array) -> float:
        return self.rn_model.predict(features)[0]

    def predict_on_array(self, features: np.array) -> np.array:
        return self.rn_model.predict(features).reshape(-1)

    def train(self, data: tuple[np.array, np.array]) -> None:
        self.rn_model.fit(data[0], data[1])

    def save(self, path_to_model: str) -> None:
        raise NotImplementedError()

    def load(self, path_to_model: str) -> None:
        raise NotImplementedError()

    def copy(self) -> any:
        new_obj = RNModel()
        new_obj.learning_rate = self.learning_rate
        new_obj.rn_model = keras.models.clone_model(self.rn_model)
        return new_obj
