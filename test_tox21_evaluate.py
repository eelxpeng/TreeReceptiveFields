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

from lib.genMask import genMask
from lib.ops import MaskedLinear
from lib.Tox21_Data import Dataset, read

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', type=str, default="checkpoint/ckpt-tree-tox.t7",
                    help='name for this run')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

x_tr_t, y_tr_t, x_valid_t, y_valid_t, x_te_t, y_te_t = read("../datasets/tox21/", target=0)
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
infeatures = 1644
outfeatures = 847
mask = torch.zeros(infeatures,outfeatures)
maskedLinear1 = MaskedLinear(infeatures, outfeatures, mask)
activate1 = nn.ReLU()

infeatures = 847
outfeatures = 847
mask = torch.zeros(infeatures,outfeatures)
maskedLinear2 = MaskedLinear(infeatures, outfeatures, mask)
activate2 = nn.ReLU()

print("=========Classify============")
fc = nn.Linear(outfeatures, num_classes)

print("==> Load from checkpoint...")
# load model
if use_cuda:
    checkpoint = torch.load(args.model)
else:
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)

maskedLinear1.load_state_dict(checkpoint['maskedLinear1'])
maskedLinear2.load_state_dict(checkpoint['maskedLinear2'])
fc.load_state_dict(checkpoint['fc'])

mask = maskedLinear1.mask.data.numpy()
print(np.sum(mask))

if use_cuda:
    maskedLinear1.cuda()
    activate1.cuda()
    maskedLinear2.cuda()
    activate2.cuda()
    fc.cuda()

criterion = nn.CrossEntropyLoss()
def test(epoch, dataloader, valid=True):
    test_loss = 0
    correct = 0
    total = 0
    alltargets = []
    allscore = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.view(inputs.size(0), -1)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        h1 = activate1(maskedLinear1(inputs))
        h2 = activate2(maskedLinear2(h1))
        outputs = fc(h2)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        alltargets.append(targets.data.cpu().numpy())
        allscore.append(outputs.data.cpu().numpy())

    alltargets = np.concatenate(alltargets)
    allscore = np.concatenate(allscore)
    auc = roc_auc_score(alltargets, allscore)
    print("#Epoch %3d: Test Loss: %.3f | Acc: %.3f%% | AUC: %.3f" % (epoch, test_loss/(batch_idx+1),
                                    100.*correct/total, auc))

test(0, validloader, valid=True)
test(0, testloader, valid=False)

print("Done.")
