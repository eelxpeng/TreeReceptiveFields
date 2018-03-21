import numpy as np
import sys
import pickle
import scipy.io

from lib.bayesnet.discretize import discretize
from lib.bayesnet.chow_liu import chow_liu
from lib.bayesnet.tree_conv import tree_conv
from lib.utils import readData

label_name = ['World', 'Sports', 'Business', 'Sci/Tech']
training_num, valid_num, test_num, vocab_size = 110000, 10000, 7600, 10000
training_file = 'dataset/agnews_training_110K_10K-TFIDF-words.txt'
valid_file = 'dataset/agnews_valid_10K_10K-TFIDF-words.txt'
test_file = 'dataset/agnews_test_7600_10K-TFIDF-words.txt'

randgen = np.random.RandomState(13)
trainX, trainY = readData(training_file, training_num, vocab_size, randgen)
validX, validY = readData(valid_file, valid_num, vocab_size)
testX, testY = readData(test_file, test_num, vocab_size)

in_features = trainX.size()[1]
out_features = 500

data = trainX.cpu().numpy()
data = (data > 0).astype(np.int32)
bins = 2
n, d = data.shape

# print("Discretizing data into %d bins for each of %d dims" % (bins, d))
# bins = [bins]*d
# data = discretize(data, bins=bins, verbose=True)

n_rv = data.shape[1]
print("===> Number of variables: %d" % n_rv)

print("===> Measure pairwise mutual_information...")
import time

def mutual_information(data):
    #bins = np.amax(data, axis=0)+1 # read levels for each variable
    # pdb.set_trace()
    bins = unique_bins(data)
        
    if len(bins) == 2:
        hist,_ = np.histogramdd(data, bins=bins[0:2]) # frequency counts

        Pxy = hist / hist.sum()# joint probability distribution over X,Y,Z
        Px = np.sum(Pxy, axis = 1) # P(X,Z)
        Py = np.sum(Pxy, axis = 0) # P(Y,Z) 

        PxPy = np.outer(Px,Py)
        Pxy += 1e-7
        PxPy += 1e-7
        MI = np.sum(Pxy * np.log(Pxy / (PxPy)))
        return round(MI,4)

def mutual_information_binary(data):
    """
    data: binary format
    """
    num, n_rv = data.shape
    frequency = np.zeros((n_rv, n_rv), dtype=np.int32)
    print("Computing single and joint counts...")
    for d in data:
        index = np.nonzero(d)[0]
        for i in index:
            frequency[i, index] += 1

    edge_list = [None]*int(n_rv*(n_rv-1)/2)
    ind = -1
    Pxy = np.zeros((2, 2), dtype=np.float32)
    Px = np.zeros(2, dtype=np.float32)
    Py = np.zeros(2, dtype=np.float32)
    for i in range(n_rv):
        for j in range(i+1, n_rv):
            Px[1] = 1.0*frequency[i, i]/num
            Px[0] = 1-Px[1]
            Py[1] = 1.0*frequency[j, j]/num
            Py[0] = 1-Py[1]
            Pxy[1, 1] = 1.0*frequency[i, j]/num
            Pxy[1, 0] = Px[1] - Pxy[1, 1]
            Pxy[0, 1] = Py[1] - Pxy[1, 1]
            Pxy[0, 0] = 1 - Px[1] - Py[1] + Pxy[1, 1]

            PxPy = np.outer(Px,Py)
            Pxy += 1e-7
            PxPy += 1e-7
            MI = np.sum(Pxy * np.log(Pxy / (PxPy)))

            ind += 1
            edge_list[ind] = (i, j, MI)

        if (i+1)%1000==0:
            print("MI computed for %d variables..." % (i+1))

    return edge_list

def unique_bins(data):
    """
    Get the unique values for each column in a dataset.
    """
    bins = np.empty(len(data.T), dtype=np.int32)
    i = 0
    for col in data.T:
        bins[i] = len(np.unique(col))
        i+=1
    return bins

start = time.time()
mutual_information_binary(data)
# for j in range(1, n_rv):
#     mutual_information(data[:,(0,j)])
end = time.time()
print(end - start)
print("Ave time", (end-start)/(n_rv-1))
