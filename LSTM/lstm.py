import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

def build_model(layers):
    model = Sequential()


    # model.add(LSTM(
    #     input_shape=(layers[1], layers[0]),
    #     output_dim=layers[2],
    #     return_sequences=False))
    # model.add(Dropout(0.05))

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=1,
        return_sequences=True))
    model.add(Dropout(0.05))
    model.add(LSTM(
        output_dim=layers[2],
        return_sequences=False))
    model.add(Dropout(0.05))
    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    rmsprop = keras.optimizers.RMSprop(lr=0.02, rho=0.9) # default lr=.001
    model.compile(loss="mse", optimizer=rmsprop)
    return model

def predict_point_by_point(model, data):
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

