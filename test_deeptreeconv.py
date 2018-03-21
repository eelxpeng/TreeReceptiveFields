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

from lib.Tox21_Data import Dataset, read
from lib.utils import readData
from lib.treeConvNet import TreeConvNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--target', type=int, default=0, metavar='N',
                    help='target task 0-11')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--name', type=str, default="tree-mnist",
                    help='name for this run')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

label_name = ['World', 'Sports', 'Business', 'Sci/Tech']
num_classes = len(label_name)
training_num, valid_num, test_num, vocab_size = 110000, 10000, 7600, 10000
training_file = 'dataset/agnews_training_110K_10K-TFIDF-words.txt'
valid_file = 'dataset/agnews_valid_10K_10K-TFIDF-words.txt'
test_file = 'dataset/agnews_test_7600_10K-TFIDF-words.txt'

randgen = np.random.RandomState(13)
trainX, trainY = readData(training_file, training_num, vocab_size, randgen)
validX, validY = readData(valid_file, valid_num, vocab_size)
testX, testY = readData(test_file, test_num, vocab_size)

input_dim = trainX.size()[1]
# 10000 -> 1000 -> 500 -> 500 -> 250 -> 250 -> 125 -> 125
kernel_stride = [(11, 10), (5, 2), (5, 1), (5, 2), (5, 1), (5, 2), (5, 1)]

net = TreeConvNet(args.name)

# net.learn_structure(trainX, validX, num_classes, kernel_stride, corrupt=0.5,
#         lr=1e-3, batch_size=args.batch_size, epochs=10)
net.net = torch.load('./checkpoint/ckpt-'+args.name+'-structure.t7')
net.fit(trainX, trainY, validX, validY, testX, testY, batch_size=args.batch_size,
    lr=args.lr, epochs=args.epochs)
print("Done.")
