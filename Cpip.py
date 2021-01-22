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
        self.LSTMbatch = 10000
        self.dbn_layers = [n, int(n*3/4), int(n*1/2)]

        # self.LSTM = lstm.build_model([self.n, self.seqlen, self.n, 1])
        self.AD = dae.DBN_AE(self.n, self.maxAE, self.FMgrace, self.ADgrace, self.dbn_layers)
        self.LSTM = lstm.build_model([self.n, self.seqlen, 1])
        self.fvs = queue.Queue(maxsize=self.LSTMbatch+self.seqlen-1)
        self.rms = queue.Queue(maxsize=self.LSTMbatch+self.seqlen-1)

    def put(self, x, rm):
        if self.fvs.full():
            self.fvs.get()
        if self.rms.full():
            self.rms.get()
        self.fvs.put(x)
        self.rms.put(rm)

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

    def gen_LSTM_data(self):
        flist = list(self.fvs.queue)
        rlist = list(self.rms.queue)
        X = np.array([flist[t:t+self.seqlen] for t in range(self.LSTMbatch)])
        Y = np.array([rlist[t-1+self.seqlen] for t in range(self.LSTMbatch)])
        return X, Y

    def get_fv_seq(self):
        flist = list(self.fvs.queue)
        rv = np.array(flist[-self.seqlen:])
        rv = rv[newaxis,:,:]
        return rv

    def train_LSTM(self, x):
        rm = self.AD.process(x)
        self.put(x, rm)
        self.LSTM_trained += 1
        if(self.LSTM_trained % self.LSTMbatch == 0):
            X_train_lstm, y_train_lstm = self.gen_LSTM_data()
            self.LSTM.fit(
                X_train_lstm,
                y_train_lstm,
                batch_size = 512,
                nb_epoch = 64,
                validation_split = 0.05,
                verbose = 0)
        return rm, 0.0

    def execute(self, x):
        rm = self.AD.process(x)
        self.put(x, rm)
        lt = lstm.predict_point_by_point(self.LSTM, self.get_fv_seq())[0]
        return rm, lt

    def save_lstm(self, model_path):
        self.LSTM.save(model_path)
        print("Model saved to path : %s" % model_path)

