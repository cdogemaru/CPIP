import os
import time
import warnings
import numpy as np
from numpy import newaxis
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self, datax, datay, num_steps=50, batch_size=64, n_feature=100, skip_step=1):
        self.datax = datax
        self.datay = datay
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.n_feature = n_feature
        self.current_idx = 0
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, self.n_feature))
        y = np.zeros((self.batch_size))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.datax):
                    self.current_idx = 0
                x[i,:,:] = self.datax[self.current_idx:self.current_idx + self.num_steps]
                y[i] = self.datay[self.current_idx + self.num_steps]
                self.current_idx += self.skip_step
            yield x, y


class lstm:
    def __init__(self, layers, n_units=50):
        self.model = Sequential()
        self.model.add(LSTM(n_units, input_shape=(layers[1], layers[0]), return_sequences=False))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(layers[2]))
        self.model.add(Activation("linear"))
        rmsprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9) # default lr=.001
        self.model.compile(loss="mse", optimizer=rmsprop)

    def fit(self, X, y):
        g = DataGenerator(X, y)
        history = self.model.fit_generator(g.generate(), X.shape[0]//(g.batch_size*g.num_steps), epochs=16, verbose=True)

    def predict(self, X):
        return self.model.predict(X)

