# coding: utf-8

from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
import Cpip
import pandas as pd
import time

X = np.load("../data/IDS-2017-feature.npy")

maxAE = 10
FMgrace = 50000
ADgrace = 150000
LSTMgrace = 100000

rms = []
lts = []
print("Running CPIP:")
start = time.time()
C = Cpip.CPIP(X.shape[1], FMgrace, ADgrace, LSTMgrace)


for i in range(X.shape[0]):
    if (i+1) % 1000 == 0:
        print(i+1)
    rm, lt = C.process(X[i, ])
    rms.append(rm)
    lts.append(lt)
stop = time.time()
print("Complete. Time elapsed: " + str(stop - start))

prms = np.array(rms[200000:])
plts = np.array(lts[200000:])
scores = np.zeros(400000)

scores[:100000] = 2 * np.exp(10 * prms[:100000])
scores[100000:] = np.exp(10 * prms[100000:]) + np.exp(10 * plts[100000:])
index = np.array(range(len(scores)))
benignSample = np.log(scores[:50000])
logProbs = norm.logsf(np.log(scores), np.mean(
    benignSample), np.std(benignSample))

fig3 = plt.figure(figsize=(12.8, 6.4))
plt.scatter(index, scores, s=4,
            c=logProbs, cmap='RdYlGn')
plt.ylim([min(scores), max(scores)+1.5])
plt.annotate('Normal Traffic', xy=(index[26000], 3), xytext=(
    index[0], max(scores)), arrowprops=dict(facecolor='black', shrink=0.005), fontsize='large')
plt.annotate('DDoS Attack Traffic', xy=(index[100000], max(scores)), xytext=(
    index[0], max(scores)+1), arrowprops=dict(facecolor='black', shrink=0.005), fontsize='large')

plt.xlabel("indexs of packets")
plt.ylabel("anomaly score")


plt.savefig("./result.png")
plt.show()
C.LSTM.save("../LSTM_demo/lstm.h5")


