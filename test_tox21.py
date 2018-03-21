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
from sklearn.metrics import roc_auc_score

from lib.genMask import genMask
from lib.ops import MaskedLinear
from lib.Tox21_Data import Dataset, read

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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

x_tr_t, y_tr_t, x_valid_t, y_valid_t, x_te_t, y_te_t = read("../datasets/tox21/", target=args.target)
trainset = Dataset(x_tr_t, y_tr_t)
validset = Dataset(x_valid_t, y_valid_t)
testset = Dataset(x_te_t, y_te_t)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(
    validset, batch_size=args.batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

num_classes = 2
train_data =  trainset.data.cpu().numpy()
train_labels =  trainset.labels.cpu().numpy()
print("- Train data shape: ", train_data.shape)
print(train_data[:2])
print(np.unique(train_labels))

# sys.exit(0)

print("Generating layer 1 units...")
mask = genMask(train_data, todiscretize=True, bins=5, kernel_size=3, stride=2)
infeatures, outfeatures = mask.shape
print("Layer 1 hidden units: ", outfeatures)
mask = torch.from_numpy(mask.astype(np.float32))
if use_cuda:
    mask = mask.cuda()
maskedLinear1 = MaskedLinear(infeatures, outfeatures, mask)
activate1 = nn.ReLU()
dropout_in = nn.Dropout()
dropout1 = nn.Dropout(p=0.2)
if use_cuda:
    maskedLinear1.cuda()
    activate1.cuda()
    dropout_in.cuda()
    dropout1.cuda()
dropout_in.train()
dropout1.train()
print("=====Autoencoding layer 1=======")
optimizer = optim.Adam(maskedLinear1.parameters(), lr=args.lr)
criterion = nn.MSELoss()

for epoch in range(args.epochs):
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.view(inputs.size(0), -1).float()
        if use_cuda:
                inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs = Variable(inputs)
        inputs_corr = dropout_in(inputs)
        hidden = dropout1(activate1(maskedLinear1(inputs_corr)))
        de = maskedLinear1.decode(hidden)
        outputs = de
        recon_loss = criterion(outputs, inputs)
        recon_loss.backward()
        optimizer.step()

    print("#Epoch %3d: Reconstruct Loss: %.3f" % (epoch, recon_loss.data[0]))

print("Projecting layer 1 output")
dropout_in.eval()
dropout1.eval()
hidden1 = []
for batch_idx, (inputs, _) in enumerate(trainloader):
    inputs = inputs.view(inputs.size(0), -1).float()
    if use_cuda:
            inputs = inputs.cuda()
    inputs = Variable(inputs)
    h = dropout1(activate1(maskedLinear1(dropout_in(inputs))))
    hidden1.append(h.data.cpu().numpy())

hidden1 = np.concatenate(hidden1, axis=0)
print("Projected data:", hidden1.shape)
print("Generating layer 2 units...")
mask = genMask(hidden1, todiscretize=True, kernel_size=3, stride=1)
infeatures, outfeatures = mask.shape
print("Layer 2 hidden units: ", outfeatures)
mask = torch.from_numpy(mask.astype(np.float32))
if use_cuda:
    mask = mask.cuda()
maskedLinear2 = MaskedLinear(infeatures, outfeatures, mask)
activate2 = nn.ReLU()
relu = nn.ReLU()
dropout2 = nn.Dropout(p=0.2)
if use_cuda:
    maskedLinear2.cuda()
    activate2.cuda()
    dropout2.cuda()
    relu.cuda()
dropout_in.eval()
dropout1.train()
dropout2.train()

print("=====Autoencoding layer 2=======")
optimizer = optim.Adam(maskedLinear2.parameters(), lr=args.lr)
criterion = nn.MSELoss()
for epoch in range(args.epochs):
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.view(inputs.size(0), -1).float()
        if use_cuda:
                inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs = Variable(inputs)
        inputs = activate1(maskedLinear1(dropout_in(inputs)))
        inputs_corr = Variable(dropout1(inputs).data, requires_grad=False)
        inputs = Variable(inputs.data, requires_grad=False)
        h = dropout2(activate2(maskedLinear2(inputs_corr)))
        outputs = relu(maskedLinear2.decode(h))
        recon_loss = criterion(outputs, inputs)
        recon_loss.backward()
        optimizer.step()

    print("#Epoch %3d: Reconstruct Loss: %.3f" % (epoch, recon_loss.data[0]))

print("=========Classify============")
fc = nn.Linear(outfeatures, 1)
sigmoid = nn.Sigmoid()
if use_cuda:
    fc.cuda()
params = [{'params': maskedLinear1.parameters()},
            {'params': maskedLinear2.parameters()}]
optimizer = optim.Adam(params, lr=args.lr)
criterion = nn.BCELoss()
best_valid_auc = 0  # best test accuracy
test_auc = 0
best_test_auc = 0
dropout_in.eval()

def test(epoch, dataloader, valid=True):
    global best_valid_auc, best_test_auc, test_auc
    test_loss = 0
    correct = 0
    total = 0
    alltargets = []
    allscore = []
    dropout1.eval()
    dropout2.eval()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.view(inputs.size(0), -1).float()
        targets = targets.view(-1, 1).float()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        h1 = dropout1(activate1(maskedLinear1(dropout_in(inputs))))
        h2 = dropout2(activate2(maskedLinear2(h1)))
        outputs = sigmoid(fc(h2))
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        total += targets.size(0)
        predicted = (outputs.data.cpu().numpy() > 0.5).astype(np.int32)
        correct += np.sum(predicted == targets.data.cpu().numpy().astype(np.int32))

        alltargets.append(targets.data.cpu().numpy())
        allscore.append(outputs.data.cpu().numpy())

    alltargets = np.concatenate(alltargets)
    allscore = np.concatenate(allscore)
    auc = roc_auc_score(alltargets, allscore)
    print("#Epoch %3d: Test Loss: %.3f | Acc: %.3f%% | AUC: %.3f" % (epoch, test_loss/(batch_idx+1),
                                    100.*correct/total, auc))
    # Save checkpoint.
    if not valid:
        test_auc = auc
    if valid:
        acc = 100.*correct/total
        if auc > best_valid_auc:
            print('Saving..')
            state = {
                'maskedLinear1': maskedLinear1.module if use_cuda else maskedLinear1,
                'maskedLinear2': maskedLinear2.module if use_cuda else maskedLinear2,
                'fc': fc.module if use_cuda else fc,
                'acc': acc,
                'auc': auc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt-'+args.name+'.t7')
            best_valid_auc = auc
            best_test_auc = test_auc

def train(epoch):
    dropout1.train()
    dropout2.train()
    train_loss = 0
    correct = 0
    total = 0
    alltargets = []
    allscore = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.view(inputs.size(0), -1).float()
        targets = targets.view(-1, 1).float()
        if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        h1 = dropout1(activate1(maskedLinear1(dropout_in(inputs))))
        h2 = dropout2(activate2(maskedLinear2(h1)))
        outputs = sigmoid(fc(h2))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        total += targets.size(0)
        predicted = (outputs.data.cpu().numpy() > 0.5).astype(np.int32)
        correct += np.sum(predicted == targets.data.cpu().numpy().astype(np.int32))

        alltargets.append(targets.data.cpu().numpy())
        allscore.append(outputs.data.cpu().numpy())

    alltargets = np.concatenate(alltargets)
    allscore = np.concatenate(allscore)
    auc = roc_auc_score(alltargets, allscore)
    print("#Epoch %3d: Train Loss: %.3f | Acc: %.3f%% | AUC: %.3f" % (epoch, train_loss/(batch_idx+1),
                                    100.*correct/total, auc))

for epoch in range(100):
    train(epoch)
    test(epoch, testloader, valid=False)
    test(epoch, validloader, valid=True)

print("Best Valid AUC=%.3f, test AUC=%.3f" % (best_valid_auc, best_test_auc))
print("Done.")
