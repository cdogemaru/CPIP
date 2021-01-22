import numpy as np
import dA as AE
from dbn.models import UnsupervisedDBN
class DBN_AE:
    def __init__(self, n, max_autoencoder_size=10, FM_grace_period=None, AD_grace_period=10000, dbn_layers=[100, 75, 50, 35, 16], learning_rate=0.1, hidden_ratio=0.75, feature_map=None):
        self.AD_grace_period = AD_grace_period
        self.FM_grace_period = FM_grace_period
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n

        
        self.n_trained = 0 # the number of training instances so far
        self.n_executed = 0 # the number of executed instances so far
        self.dbn_batch = 10000
        self.dbn_layers = dbn_layers
        self.AE_dim = dbn_layers[-1]
        self.__createDBN__()
        self.__createAE__()
        self.fvs = []
        print("Deep Belief Network: train-mode, Auto-Encoder: off-mode")

    def __createAE__(self):
        params = AE.dA_params(self.AE_dim, n_hidden=0, lr=self.lr,
                corruption_level=0, gracePeriod=0, hiddenRatio=self.hr)
        self.AE = AE.dA(params)

    def __createDBN__(self):
        self.FM = UnsupervisedDBN(hidden_layers_structure=self.dbn_layers,
                                         batch_size=512,
                                         learning_rate_rbm=0.3,
                                         n_epochs_rbm=64,
                                         activation_function='sigmoid',
                                         verbose=False)

    def process(self,x):
        if self.n_trained < self.FM_grace_period + self.AD_grace_period:
            self.train(x)
            return 0.0
        else:
            return self.execute(x)

    def train_FM(self, x):
        self.fvs.append(x)
        if len(self.fvs) == self.dbn_batch:
            xx = np.array(self.fvs)
            self.FM.fit(xx)
            self.fvs.clear()

    def train(self,x):
        if self.n_trained < self.FM_grace_period:
            self.train_FM(x)
        else:
            S_l1 = self.FM.transform(x)
            self.AE.train(S_l1)
        self.n_trained += 1
        if self.n_trained == self.AD_grace_period+self.FM_grace_period:
                print("Deep Belief Network: execute-mode, Auto-Encoder: train-mode")

    def execute(self,x):
        self.n_executed += 1
        S_l1 = self.FM.transform(x)
        return self.AE.execute(S_l1)
