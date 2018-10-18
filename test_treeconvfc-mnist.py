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
import sys
import pickle
from sklearn.metrics import roc_auc_score
import time
import pdb

from lib.Tox21_Data import read
from lib.utils import readData, Dataset, TextDataset
from lib.treeConvNetFC import TreeConvNetFC
from lib.maskedDAEwithFC import MaskedDenoisingAutoencoderFC
from lib.mnist import load_mnist

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--dataset', type=str, default="agnews",
                    help='dataset')
parser.add_argument('--target', type=int, default=0, metavar='N',
                    help='target task 0-11')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--name', type=str, default="tree-mnist",
                    help='name for this run')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

kernel_stride = [(15, 1), (5, 2), (5, 1)]
fcwidths = [100, 50, 50]

print("Running Dataset %s with [lr=%f, epochs=%d, batch_size=%d]"
        % (args.dataset, args.lr, args.epochs, args.batch_size))
print("kernel_stride: ", kernel_stride)
print("fcwidths: ", fcwidths)

start_time = time.time()
num_classes = 10
X, Y = load_mnist(dataset="training", path="dataset/mnist/")
testX, testY = load_mnist(dataset="testing", path="dataset/mnist/")
X = (X.astype(np.float32)/255).reshape(-1, 784)
Y = Y.astype(np.int)
num_train = 50000
trainX = torch.from_numpy(X[:num_train])
trainY = torch.from_numpy(Y[:num_train]).type(torch.LongTensor)
validX = torch.from_numpy(X[num_train:])
validY = torch.from_numpy(Y[num_train:]).type(torch.LongTensor)
testX = torch.from_numpy((testX.astype(np.float32)/255).reshape(-1, 784))
testY = torch.from_numpy(testY.astype(np.float32)).type(torch.LongTensor)

end_time = time.time()
print("reading data cost: ", end_time - start_time)

input_dim = trainX.shape[1]
# input_dim = trainset.shape[1]

net = TreeConvNetFC(args.name)

start_time = time.time()
# net.learn_structure(trainX, validX, num_classes, kernel_stride, fcwidths, corrupt=0.5,
#         lr=1e-3, batch_size=args.batch_size, epochs=10)
# net.learn_structure(trainX, validX, num_classes, kernel_stride, fcwidths, corrupt=0.5,
#         lr=1e-3, batch_size=args.batch_size, epochs=10)
net.learn_structure(trainX, validX, num_classes, kernel_stride, fcwidths, corrupt=0.5,
        lr=1e-3, batch_size=args.batch_size, epochs=10, loss_type="cross-entropy")
end_time = time.time()
print("learning structure cost: ", end_time - start_time)

trainset = Dataset(trainX, trainY)
validset = Dataset(validX, validY)
testset = Dataset(testX, testY)

start_time = time.time()
net.fit(trainset, validset, testset, batch_size=args.batch_size,
    lr=args.lr, epochs=args.epochs)
end_time = time.time()
print("finetuning cost: ", end_time - start_time)
print("Done.")