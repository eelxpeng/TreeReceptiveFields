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

from lib.Tox21_Data import Dataset, read
from lib.utils import readData
from lib.treeConvNetFC import TreeConvNetFC
from lib.maskedDAEwithFC import MaskedDenoisingAutoencoderFC
from lib.utils import Dataset

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

if args.dataset=="agnews":
    label_name = ['World', 'Sports', 'Business', 'Sci/Tech']
    training_num, valid_num, test_num, vocab_size = 110000, 10000, 7600, 10000
    training_file = 'dataset/agnews_training_110K_10K-TFIDF-words.txt'
    valid_file = 'dataset/agnews_valid_10K_10K-TFIDF-words.txt'
    test_file = 'dataset/agnews_test_7600_10K-TFIDF-words.txt'
    # 10000 -> 1000 -> 500 -> 250
    kernel_stride = [(9, 10), (5, 2), (5, 2)]
    fcwidths = [100, 50, 25]

elif args.dataset=="dbpedia":
    label_name = ['Company','EducationalInstitution','Artist','Athlete','OfficeHolder','MeanOfTransportation','Building','NaturalPlace','Village','Animal','Plant','Album','Film','WrittenWork']
    training_num, valid_num, test_num, vocab_size = 549990, 10010, 70000, 10000
    training_file = 'dataset/dbpedia_training_549990_10K-frequent-words.txt'
    valid_file = 'dataset/dbpedia_valid_10010_10K-frequent-words.txt'
    test_file = 'dataset/dbpedia_test_70K_10K-frequent-words.txt'
    # 10000 -> 1000 -> 500 -> 250
    kernel_stride = [(9, 10), (5, 2), (5, 2)]
    fcwidths = [100, 50, 50]
    # kernel_stride = [(10, 15), (5, 1), (5, 2)]
    # fcwidths = [100, 100, 50]

elif args.dataset=="sogounews":
    label_name = ['sports','finance','entertainment','automobile','technology']
    training_num, valid_num, test_num, vocab_size = 440000, 10000, 60000, 10000
    training_file = 'dataset/sogounews_training_440K_10K-frequent-words.txt'
    valid_file = 'dataset/sogounews_valid_10K_10K-frequent-words.txt'
    test_file = 'dataset/sogounews_test_60K_10K-frequent-words.txt'
    # 10000 -> 1000 -> 500 -> 250
    # kernel_stride = [(9, 10), (5, 2), (5, 2)]
    # fcwidths = [100, 50, 50]
    
    kernel_stride = [(6, 5), (5, 4), (5, 2)]
    fcwidths = [100, 50, 25]

elif args.dataset=="yelp":
    label_name = ['1','2','3','4','5']
    training_num, valid_num, test_num, vocab_size = 640000, 10000, 50000, 10000
    training_file = 'dataset/yelpfull_training_640K_10K-frequent-words.txt'
    valid_file = 'dataset/yelpfull_valid_10K_10K-frequent-words.txt'
    test_file = 'dataset/yelpfull_test_50K_10K-frequent-words.txt' 
    # 10000 -> 1000 -> 500 -> 250
    kernel_stride = [(9, 10), (5, 2), (5, 2)]
    fcwidths = [100, 50, 50]

elif args.dataset=="yahoo":
    label_name = ["Society & Culture","Science & Mathematics","Health","Education & Reference","Computers & Internet","Sports","Business & Finance","Entertainment & Music","Family & Relationships","Politics & Government"]
    training_num, valid_num, test_num, vocab_size = 1390000, 10000, 60000, 10000
    training_file = 'dataset/yahoo_training_1.39M_10K-frequent-words.txt'
    valid_file = 'dataset/yahoo_valid_10K_10K-frequent-words.txt'
    test_file = 'dataset/yahoo_test_60K_10K-frequent-words.txt'
    # 10000 -> 1000 -> 500 -> 250
    # kernel_stride = [(9, 10), (5, 2), (5, 2)]
    # fcwidths = [100, 50, 50]
    # 10000 -> 1000 -> 500 -> 250
    # kernel_stride = [(9, 10), (5, 2), (5, 2)]
    # fcwidths = [200, 100, 50]
    # 10000 -> 2000 -> 500 -> 250
    # kernel_stride = [(6, 5), (5, 4), (5, 2)]
    # fcwidths = [200, 100, 50]
    # 10000 -> 500 -> 500 -> 250
    # kernel_stride = [(10, 15), (5, 1), (5, 2)]
    # fcwidths = [100, 100, 50]
    # 10000 -> 1000 -> 500 -> 500
    # kernel_stride = [(9, 10), (5, 2), (5, 1)]
    # fcwidths = [100, 100, 100]
    # 10000 -> 1000 -> 500 -> 500
    # kernel_stride = [(10, 15), (5, 1), (5, 1)]
    # fcwidths = [500, 500, 300]

    kernel_stride = [(6, 5), (5, 4), (5, 2)]
    fcwidths = [100, 50, 25]


print("Running Dataset %s with [lr=%f, epochs=%d, batch_size=%d]"
        % (args.dataset, args.lr, args.epochs, args.batch_size))
print("kernel_stride: ", kernel_stride)
print("fcwidths: ", fcwidths)

start_time = time.time()
num_classes = len(label_name)
randgen = np.random.RandomState(13)
trainX, trainY = readData(training_file, training_num, vocab_size, randgen)
validX, validY = readData(valid_file, valid_num, vocab_size)
testX, testY = readData(test_file, test_num, vocab_size)

end_time = time.time()
print("reading data cost: ", end_time - start_time)

# preprocess, normalize each dimension to be [0, 1] for cross-entropy loss
train_max = torch.max(trainX, dim=0, keepdim=True)[0]
valid_max = torch.max(validX, dim=0, keepdim=True)[0]
test_max = torch.max(testX, dim=0, keepdim=True)[0]
print(train_max.size())
print(valid_max.size())
print(test_max.size())
x_max = torch.max(torch.cat((train_max, valid_max, test_max), 0), dim=0, keepdim=True)[0]
trainX.div_(x_max)
validX.div_(x_max)
testX.div_(x_max)

trainset = Dataset(trainX, trainY)
validset = Dataset(validX, validY)
testset = Dataset(testX, testY)

input_dim = trainX.size()[1]

net = TreeConvNetFC(args.name)

start_time = time.time()
net.learn_structure(trainX, validX, num_classes, kernel_stride, fcwidths, corrupt=0.5,
        lr=1e-3, batch_size=args.batch_size, epochs=10, loss_type="cross-entropy")
end_time = time.time()
print("learning structure cost: ", end_time - start_time)
# net.net = torch.load('./checkpoint/ckpt-'+args.name+'-structure.t7')
# use skeleton only, initialize weight randomly
# for l in net.net:
#     if isinstance(l, MaskedDenoisingAutoencoder):
#         l.reset_parameters()
#     if isinstance(l, nn.Dropout):
#         l.p = 0.5

start_time = time.time()
net.fit(trainset, validset, testset, batch_size=args.batch_size,
    lr=args.lr, epochs=args.epochs)
end_time = time.time()
print("finetuning cost: ", end_time - start_time)
print("Done.")