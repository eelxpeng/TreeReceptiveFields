import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import os
import argparse
import logging
import scipy.io as sio
from sklearn.metrics import roc_auc_score
import copy
from torch.optim.lr_scheduler import LambdaLR, StepLR
from lib.utils import *
from lib.ops import *

# Training settings
parser = argparse.ArgumentParser(description='DNN parameters')
parser.add_argument('--batch-size', type=int, default=1000, 
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, 
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, 
                    help='number of epochs to train (default: 10)')
parser.add_argument('--dataset', type=str, default="tox",
                    help='dataset')
parser.add_argument('--target', type=int, default=0, 
                    help='target task 0-11')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--num-layer', type=int, default=1, 
                    help='num hidden layers')
parser.add_argument('--shape', type=int, default=0, 
                    help='0 for rectange, 1 for conic')
parser.add_argument('--num-neuron', type=int, default=512, 
                    help='num neurons for first hidden layer')
parser.add_argument('--wd', type=float, default=0.0005, 
                    help='pruning sparsity')
parser.add_argument('--name', type=str, default="tox-1-512",
                    help='name for this run')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
batchsize = args.batch_size

dropout = 0.5
class FNN(nn.Module):
    def __init__(self, dims, dropout=0.5):
        super(FNN, self).__init__()

        self.fc = []
        for i in range(len(dims)-2):
            self.fc.append( nn.Linear(dims[i], dims[i+1]) )
            self.fc.append( nn.ReLU() )
            self.fc.append( nn.Dropout(p=dropout) )
        self.fc.append( nn.Linear(dims[-2], dims[-1]) )
        
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        return self.fc(x)

fnn = torch.load("./checkpoint/ckpt-sparsel1-agnews-l3-con-1024.t7")

def net_histogram(fnn):
    weights = []
    for l in range(len(fnn.fc)):
        if isinstance(fnn.fc[l], nn.Linear):
            weight = np.abs(fnn.fc[l].weight.data.cpu().numpy())
            weights.append(weight.flatten())
    weights = np.concatenate(weights)
    print(np.histogram(weights))

def check_sparsity(fnn, thresh=0.0):
    nonzeros = 0
    count = 0
    for l in range(len(fnn.fc)):
        if isinstance(fnn.fc[l], nn.Linear):
            weight = np.abs(fnn.fc[l].weight.data.cpu().numpy())
            count += weight.shape[0]*weight.shape[1]
            nonzeros += np.sum(weight>thresh)
    return 1.0*nonzeros/count

net_histogram(fnn)
print("Sparsity: %f" % check_sparsity(fnn, thresh=1e-3))

