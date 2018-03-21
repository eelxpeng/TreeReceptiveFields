import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import scipy.io
import os
import os.path
import pickle
import sys

from lib.utils import readData, TextDataset
from lib.genMask import genMask, genMaskMIIslands
from lib.maskedDAE import MaskedDenoisingAutoencoder
from lib.maskedDAEwithFC import MaskedDenoisingAutoencoderFC

def makeLayer(data_x, valid_x, corrupt=0.5, kernel_size=3, stride=2, 
    lr=0.01, batch_size=128, epochs=100, loss_type="mse", thresh=0):
    """
    data_x, data_y: FloatTensor
    """
    use_cuda = torch.cuda.is_available()
    print("Generating layer units...")
    # maskfile = "mask-agnews.pkl"
    # if os.path.exists(maskfile):
    #     mask = pickle.load(open(maskfile,"rb"))
    # else:
    #     mask = genMask(data_x.cpu().numpy(), kernel_size=kernel_size, stride=stride)
    #     pickle.dump(mask, open(maskfile, "wb"))
    mask = genMask(data_x.cpu().numpy(), thresh=thresh, kernel_size=kernel_size, stride=stride)
    infeatures, outfeatures = mask.shape
    print("Layer hidden units: %d, Density: %f" % (outfeatures, 1.0*np.sum(mask)/(mask.shape[0]*mask.shape[1])))
    mask = torch.from_numpy(mask.astype(np.float32).T)
    dae = MaskedDenoisingAutoencoder(infeatures, outfeatures, mask)
    dae.fit(data_x, valid_x, lr=lr, batch_size=batch_size, num_epochs=epochs, corrupt=corrupt, loss_type=loss_type)

    return dae

def makeLayerMaskedDAEFC(data_x, valid_x, corrupt=0.5, kernel_size=3, stride=2, fcwidth=10,
    lr=0.01, batch_size=128, epochs=100, loss_type="mse"):
    """
    data_x, data_y: FloatTensor
    """
    use_cuda = torch.cuda.is_available()
    print("Generating layer units...")
    # maskfile = "mask-agnews.pkl"
    # if os.path.exists(maskfile):
    #     mask = pickle.load(open(maskfile,"rb"))
    # else:
    #     mask = genMask(data_x.cpu().numpy(), kernel_size=kernel_size, stride=stride)
    #     pickle.dump(mask, open(maskfile, "wb"))
    if isinstance(data_x, TextDataset):
        mask = genMask(data_x, kernel_size=kernel_size, stride=stride)    
    else:
        mask = genMask(data_x.cpu().numpy(), kernel_size=kernel_size, stride=stride)
    infeatures, outfeatures = mask.shape
    print("Layer hidden units: %d, Density: %f" % (outfeatures, 1.0*np.sum(mask)/(mask.shape[0]*mask.shape[1])))
    mask = torch.from_numpy(mask.astype(np.float32).T)
    dae = MaskedDenoisingAutoencoderFC(infeatures, outfeatures, fcwidth, mask)
    dae.fit(data_x, valid_x, lr=lr, batch_size=batch_size, num_epochs=epochs, corrupt=corrupt, loss_type=loss_type)

    return dae

def makeLayerMaskedDAEFCMI(data_x, valid_x, corrupt=0.5, kernel_size=3, stride=2, fcwidth=10,
    lr=0.01, batch_size=128, epochs=100, loss_type="mse"):
    """
    data_x, data_y: FloatTensor
    """
    use_cuda = torch.cuda.is_available()
    print("Generating layer units...")
    mask = genMaskMIIslands(data_x.cpu().numpy(), kernel_size=kernel_size, stride=stride)
    infeatures, outfeatures = mask.shape
    print("Layer hidden units: %d, Density: %f" % (outfeatures, 1.0*np.sum(mask)/(mask.shape[0]*mask.shape[1])))
    mask = torch.from_numpy(mask.astype(np.float32).T)
    dae = MaskedDenoisingAutoencoderFC(infeatures, outfeatures, fcwidth, mask)
    dae.fit(data_x, valid_x, lr=lr, batch_size=batch_size, num_epochs=epochs, corrupt=corrupt, loss_type=loss_type)

    return dae


if __name__ == "__main__":
    # from lib.Tox21_Data import read
    # x_tr_t, y_tr_t, x_valid_t, y_valid_t, x_te_t, y_te_t = read("./dataset/tox21/", target=0)
    # makeLayer(x_tr_t, y_tr_t, x_valid_t, y_valid_t, bins=2, lr=1e-4, epochs=10)
    
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
    makeLayer(trainX, validX, kernel_size=21, stride=20, lr=1e-3, epochs=10)

