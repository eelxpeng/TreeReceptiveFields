import numpy as np
import sys
import pickle
import scipy.io

from lib.bayesnet.discretize import discretize
from lib.bayesnet.chow_liu import chow_liu
from lib.bayesnet.tree_conv import tree_conv

# load 20newsgroup data
data_path = "dataset/20news/train_data.txt"
vocab_path = "dataset/20news/vocab.txt"

def loadData(path):
    data = []
    for line in open(path):
        arr = line.strip().split()
        d = [int(x) for x in arr]
        data.append(d)
    return np.array(data)

def loadVocab(path):
    vocab = []
    for line in open(path):
        w = line.strip().split()[0]
        vocab.append(w)
    return vocab

data = loadData(data_path)
vocab = loadVocab(vocab_path)

n, d = data.shape
thresh = 0
data = (data>thresh).astype(np.int8)
mst = chow_liu(data)

print("Maximum Spanning Tree")
with open("visual/mst.txt", "w") as fid:
    for key, value in mst.items():
        w = vocab[key]
        w_adj = [vocab[v] for v in value]
        fid.write("%s: %s\n" % (w, " ".join(w_adj)))

print("===> Running tree_conv to generate neiborhoods...")
islands = tree_conv(mst, 0, kernel_size=2, stride=3)

with open("visual/islands.txt", "w") as fid:
    for h in islands:
        w = vocab[h]
        w_obs = [vocab[v] for v in islands[h]]
        fid.write("%s: %s\n" % (w, " ".join(w_obs)))
print("DONE.")
