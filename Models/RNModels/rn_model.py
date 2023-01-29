import keras.models
import numpy as np
from Models.generic_ml_model import GenericMLModel
from keras import backend,regularizers
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam


class RNModel(GenericMLModel):
    def __init__(self, rn_model: Model = None):
        super().__init__()
        self.rn_model = rn_model
        self.learning_rate = 0.001

    def init_for_training(self, input_shape: int) -> None:
        input_layer = Input(shape=input_shape) ## 88

        dense_layer_1 = Dense(1024, kernel_initializer='lecun_normal', activation="selu",bias_initializer='zeros',\
             activity_regularizer = regularizers.L2(0.01),\
                bias_regularizer = regularizers.L2(0.01))\
                    (input_layer)
        dense_layer_2 = Dense(512, kernel_initializer='lecun_normal', activation="selu",bias_initializer='zeros',\
             activity_regularizer = regularizers.L2(0.01),\
                bias_regularizer = regularizers.L2(0.01))\
                    (dense_layer_1)
        dense_layer_3 = Dense(64,kernel_initializer='lecun_normal', activation="selu",bias_initializer='zeros',\
             activity_regularizer = regularizers.L2(0.01),\
                bias_regularizer = regularizers.L2(0.01))\
                    (dense_layer_2)

        output_layer = Dense(1, activation = "relu")(dense_layer_3)

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
        new_obj = RNModel()
        new_obj.learning_rate = self.learning_rate
        new_obj.rn_model = keras.models.clone_model(self.rn_model)
        new_obj.rn_model.set_weights(self.rn_model.get_weights())
        return new_obj
