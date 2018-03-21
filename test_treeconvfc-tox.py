import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import FloatTensor, LongTensor

import numpy as np
from scipy import io
import pandas as pd
import os
import sys
import pickle
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

from lib.Tox21_Data import Dataset, read
from lib.utils import readData
from lib.treeConvNetFC import TreeConvNetFC
from lib.maskedDAE import MaskedDenoisingAutoencoder

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--dataset', type=str, default="tox",
                    help='dataset')
parser.add_argument('--target', type=int, default=0, metavar='N',
                    help='target task 0-11')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--name', type=str, default="tree-mnist",
                    help='name for this run')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

def fit(net, trainX, trainY, validX, validY, testX, testY,
    batch_size=256, lr=0.01, epochs=10, name="tox"):
    print("=========Classify============")
    use_cuda = torch.cuda.is_available()
    trainset = Dataset(trainX, trainY)
    validset = Dataset(validX, validY)
    testset = Dataset(testX, testY)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(net)
    if use_cuda:
        net.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = LambdaLR( optimizer, lr_lambda=lambda epoch:1.0/np.sqrt(epoch+1) )
    criterion = nn.CrossEntropyLoss()
    best_valid_acc = 0  # best test accuracy
    best_test_acc = 0

    def test(net, epoch, dataloader):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        alloutputs = []
        alltargets = []
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            total += targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets.data).sum()

            alloutputs += [F.softmax(outputs)[:,1].data.cpu().numpy()]
            alltargets += [targets.data.cpu().numpy()]

        alloutputs = np.concatenate(alloutputs, axis=0)
        alltargets = np.concatenate(alltargets, axis=0)

        acc = 100.*correct/total
        auc_score = roc_auc_score(alltargets, alloutputs)
        print("#Epoch %3d: Test Loss: %.3f | AUC: %.4f" % (epoch, test_loss/(batch_idx+1), 
            auc_score))
        
        return auc_score

    def train(net, epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            total += targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets.data).sum()

            # auc_score = roc_auc_score(targets.data.cpu().numpy(), F.softmax(outputs)[:,1].data.cpu().numpy())

        print("#Epoch %3d: Train Loss: %.3f" % (epoch, train_loss/(batch_idx+1)))

    for epoch in range(epochs):
        scheduler.step()
        train(net, epoch)
        accValid = test(net, epoch, validloader)
        accTest = test(net, epoch, testloader)

        if accValid > best_valid_acc:
            print('Saving..')
            state = {
                'net': net,
                'acc': accValid,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt-'+name+'.t7')
            best_valid_acc = accValid
            best_test_acc = accTest

    print("\nBest Valid AUC=%.4f, test AUC=%.4f" % (best_valid_acc, best_test_acc))
    return (best_valid_acc, best_test_acc)

label_name = ['0','1']
target = 0
trainX, trainY, validX, validY, testX, testY = read("dataset/tox21/", target=target, valid_num=500)
training_num, test_num, vocab_size = trainX.size()[0], testX.size()[0], trainX.size()[1]

print("Running Dataset %s with [lr=%f, epochs=%d, batch_size=%d]"
        % (args.dataset, args.lr, args.epochs, args.batch_size))

num_classes = len(label_name)

input_dim = trainX.size()[1]
# 1644 -> 800 -> 400 -> 200
kernel_stride = [(5, 2), (5, 2), (5, 2)]
fcwidths = [80, 40, 20]

net = TreeConvNetFC(args.name)

binary_thresh = torch.from_numpy(np.mean(trainX.numpy(), axis=0, keepdims=True))
# net.learn_structure(trainX, validX, num_classes, kernel_stride, corrupt=0.5,
#         lr=1e-2, batch_size=args.batch_size, epochs=10, loss_type="cross-entropy", thresh=binary_thresh)

trainX = trainX - binary_thresh
validX = validX - binary_thresh

net.learn_structure(trainX, validX, num_classes, kernel_stride, fcwidths, corrupt=0.5,
        lr=1e-3, batch_size=args.batch_size, epochs=10)

# net.net = torch.load('./checkpoint/ckpt-'+args.name+'-structure.t7')
# use skeleton only, initialize weight randomly
# for l in net.net:
#     if isinstance(l, MaskedDenoisingAutoencoder):
#         l.reset_parameters()
#     if isinstance(l, nn.Dropout):
#         l.p = 0.5

valid_aucs = []
test_aucs = []
for target in range(12):
    print("Experiment with Target=%d" % target)
    trainX, trainY, validX, validY, testX, testY = read("dataset/tox21/", target=target, valid_num=500)
    training_num, test_num, vocab_size = trainX.size()[0], testX.size()[0], trainX.size()[1]
    
    trainX = trainX - binary_thresh
    validX = validX - binary_thresh
    testX = testX - binary_thresh

    valid_auc, test_auc = fit(net.net, trainX, trainY, validX, validY, testX, testY, batch_size=args.batch_size,
                                lr=args.lr, epochs=args.epochs, name=args.name+"-"+str(target))
    valid_aucs.append(valid_auc)
    test_aucs.append(test_auc)

print("Average valid AUC: ", np.mean(valid_aucs))
print("Average test AUC: ", np.mean(test_aucs))
print("Done.")