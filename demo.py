from FeatureExtractor import FE
from Nomalizor import Normalizor
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")
import matplotlib 
matplotlib.use("AGG")
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
import Cpip
import pandas as pd
import time
import zipfile

if not os.path.exists("./data/IDS2017.pcap"):
    with zipfile.ZipFile("./data/IDS2017.zip", "r") as zipf:
        zipf.extractall(path="./data")

fe = FE("./data/IDS2017.pcap")
X_unormalized = fe.feature_extract()

n = Normalizor()
n.fit(X_unormalized)
X = n.normalize(X_unormalized)

maxAE = 10
FMgrace = 5000
ADgrace = 195000
LSTMgrace = 100000

rms = []
lts = []

print("Running CPIP:")
start = time.time()
C = Cpip.CPIP(X.shape[1], FMgrace, ADgrace, LSTMgrace)


for i in range(X.shape[0]):
    if (i+1) % 10000 == 0:
        print(i+1)
    rm, lt = C.process(X[i, ])
    rms.append(rm)
    lts.append(lt)
stop = time.time()
print("Complete. Time elapsed: " + str(stop - start))

prms = np.array(rms[200000:])
plts = np.array(lts[200000:])
scores = np.zeros(250000)

scores[:100000] = 2 * np.exp(10 * prms[:100000])
scores[100000:] = np.exp(10 * prms[100000:]) + 2 * np.exp(10 * plts[100000:])
index = np.array(range(len(scores)))

np.save("scores.npy", scores)

benignSample = np.log(scores[:50000])
logProbs = norm.logsf(np.log(scores), np.mean(
    benignSample), np.std(benignSample))

fig = plt.figure(figsize=(12.8, 6.4))
plt.scatter(index, scores, s=4,
            c=logProbs, cmap='RdYlGn')
plt.ylim([min(scores), max(scores)+1.5])
plt.annotate('Normal Traffic', xy=(index[32000], 3), xytext=(
    index[0], max(scores)-1), arrowprops=dict(facecolor='black', shrink=0.005), fontsize='large')
plt.annotate('DDoS Attack Traffic', xy=(index[100000], max(scores)), xytext=(
    index[0], max(scores)+1), arrowprops=dict(facecolor='black', shrink=0.005), fontsize='large')

plt.xlabel("indexs of packets")
plt.ylabel("anomaly score")

plt.savefig("./result.png")