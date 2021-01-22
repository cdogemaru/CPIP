import sys
import numpy
from utils import *
import json

class dA_params:
    def __init__(self,n_visible = 5, n_hidden = 3, lr=0.001, corruption_level=0.0, gracePeriod = 10000, hiddenRatio=None):
        self.n_visible = n_visible# num of units in visible (input) layer
        self.n_hidden = n_hidden# num of units in hidden layer
        self.lr = lr
        self.corruption_level = corruption_level
        self.gracePeriod = gracePeriod
        self.hiddenRatio = hiddenRatio

class dA:
    def __init__(self, params):
        self.params = params

        if self.params.hiddenRatio is not None:
            self.params.n_hidden = int(numpy.ceil(self.params.n_visible*self.params.hiddenRatio))

        # for 0-1 normlaization
        self.norm_max = numpy.ones((self.params.n_visible,)) * -numpy.Inf
        self.norm_min = numpy.ones((self.params.n_visible,)) * numpy.Inf
        self.n = 0

        self.rng = numpy.random.RandomState(1234)

        a = 1. / self.params.n_visible
        self.W = numpy.array(self.rng.uniform(  # initialize W uniformly
            low=-a,
            high=a,
            size=(self.params.n_visible, self.params.n_hidden)))

        self.hbias = numpy.zeros(self.params.n_hidden)  # initialize h bias 0
        self.vbias = numpy.zeros(self.params.n_visible)  # initialize v bias 0
        self.W_prime = self.W.T


    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1

        return self.rng.binomial(size=input.shape,
                                 n=1,
                                 p=1 - corruption_level) * input

    # Encode
    def get_hidden_values(self, input):
        return sigmoid(numpy.dot(input, self.W) + self.hbias)

    # Decode
    def get_reconstructed_input(self, hidden):
        return sigmoid(numpy.dot(hidden, self.W_prime) + self.vbias)

    def train(self, x):
        self.n = self.n + 1

        if self.params.corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x, self.params.corruption_level)
        else:
            tilde_x = x
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W = numpy.outer(tilde_x.T, L_h1) + numpy.outer(L_h2.T, y)

        self.W += self.params.lr * L_W
        self.hbias += self.params.lr * numpy.mean(L_hbias, axis=0)
        self.vbias += self.params.lr * numpy.mean(L_vbias, axis=0)
        return numpy.sqrt(numpy.mean(L_h2**2)) #the RMSE reconstruction error during training


    def reconstruct(self, x):
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_input(y)
        return z

    def execute(self, x): #returns MSE of the reconstruction of x
        if self.n < self.params.gracePeriod:
            return 0.0
        else:
            z = self.reconstruct(x)
            rmse = numpy.sqrt(((x - z) ** 2).mean()) #MSE
            return rmse


    def inGrace(self):
        return self.n < self.params.gracePeriod
