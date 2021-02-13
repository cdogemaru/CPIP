import numpy as np
from numpy import newaxis
import pandas as pd
import time
import DBN_AE as dae
import LSTM.lstm as lstm
import queue
class CPIP:
    def __init__(self, n, FMgrace, ADgrace, LSTMgrace):
        self.FMgrace = FMgrace
        self.ADgrace = ADgrace
        self.LSTMgrace = LSTMgrace
        self.n = n

        self.AD_trained = 0
        self.LSTM_trained = 0
        self.maxAE = 10
        self.seqlen = 50
        self.dbn_layers = [n, int(n*3/4), int(n*1/2)]

        self.AD = dae.DBN_AE(self.n, self.maxAE, self.FMgrace, self.ADgrace, self.dbn_layers)
        self.LSTM = lstm.lstm([self.n, self.seqlen, 1])
        self.fvs = []
        self.rms = [] 

    def put(self, x, rm):
        self.fvs.append(x)
        self.rms.append(rm)

    def process(self, x):
        if(self.AD_trained < self.FMgrace + self.ADgrace):
            self.AD.process(x)
            self.AD_trained += 1
            self.put(x, 0.0)
            if(self.AD_trained == self.FMgrace + self.ADgrace):
                print("DBN_AE training finished. Start training LSTM.")
            return 0.0, 0.0
        elif(self.LSTM_trained < self.LSTMgrace):
            v = self.train_LSTM(x)
            if(self.LSTM_trained == self.LSTMgrace):
                print("DBN_AE and LSTM training finished. Start executing phase.")
            return v
        else:
            return self.execute(x)

    def get_fv_seq(self):
        flist = self.fvs
        rv = np.array(flist[-self.seqlen:])
        rv = rv[newaxis,:,:]
        return rv

    def gen_LSTM_data(self):
        X_train_lstm = np.array(self.fvs[-self.LSTMgrace:])
        y_train_lstm = np.array(self.rms[-self.LSTMgrace:])
        return X_train_lstm, y_train_lstm

    def train_LSTM(self, x):
        rm = self.AD.process(x)
        self.put(x, rm)
        self.LSTM_trained += 1
        if(self.LSTM_trained == self.LSTMgrace):
            X_train_lstm, y_train_lstm = self.gen_LSTM_data()
            self.LSTM.fit(
                X_train_lstm,
                y_train_lstm)
        return rm, 0.0

    def execute(self, x):
        rm = self.AD.process(x)
        self.put(x, rm)
        lt = self.LSTM.predict(self.get_fv_seq())[0][0]
        return rm, lt



