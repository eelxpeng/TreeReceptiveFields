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

from lib.genMask import genMask
from lib.ops import MaskedLinear

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--name', type=str, default="tree-mnist",
                    help='name for this run')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

transform_train = transforms.Compose([
                       transforms.ToTensor()
                    ])

transform_test = transforms.Compose([
                       transforms.ToTensor()
                   ])

trainset = datasets.MNIST('../datasets', train=True, download=True,
                   transform=transform_train)
testset = datasets.MNIST('../datasets', train=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

num_classes = 10
train_data =  trainset.train_data.cpu().numpy().reshape((-1, 784))
print("- Train data shape: ", train_data.shape)

print("Generating layer 1 units...")
mask = genMask(train_data, todiscretize=True, bins=5, kernel_size=3, stride=3)
# mask = np.load("layer1-mask.npy")
infeatures, outfeatures = mask.shape
np.save("layer1-mask", mask)
print("Layer 1 hidden units: ", outfeatures)
mask = torch.from_numpy(mask.astype(np.float32))
if use_cuda:
    mask = mask.cuda()
maskedLinear1 = MaskedLinear(infeatures, outfeatures, mask)
activate1 = nn.Sigmoid()
sigmoid = nn.Sigmoid()
if use_cuda:
    maskedLinear1.cuda()
    activate1.cuda()
    sigmoid.cuda()

print("=====Autoencoding layer 1=======")
optimizer = optim.Adam(maskedLinear1.parameters(), lr=args.lr)
criterion = nn.BCELoss()

for epoch in range(args.epochs):
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.view(inputs.size(0), -1)
        if use_cuda:
                inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs = Variable(inputs)
        hidden = activate1(maskedLinear1(inputs))
        de = maskedLinear1.decode(hidden)
        outputs = sigmoid(de)
        recon_loss = criterion(outputs, inputs)
        recon_loss.backward()
        optimizer.step()

    print("#Epoch %3d: Reconstruct Loss: %.3f" % (epoch, recon_loss.data[0]))

print("Projecting layer 1 output")
hidden1 = []
for batch_idx, (inputs, _) in enumerate(trainloader):
    inputs = inputs.view(inputs.size(0), -1)
    if use_cuda:
            inputs = inputs.cuda()
    inputs = Variable(inputs)
    h = activate1(maskedLinear1(inputs))
    hidden1.append(h.data.cpu().numpy())

hidden1 = np.concatenate(hidden1, axis=0)
scipy.io.savemat("hidden1.mat", {"hidden1": hidden1})
print("Projected data:", hidden1.shape)
print("Generating layer 2 units...")
mask = genMask(hidden1, todiscretize=True, kernel_size=3, stride=3)
infeatures, outfeatures = mask.shape
np.save("layer2-mask", mask)
print("Layer 2 hidden units: ", outfeatures)
mask = torch.from_numpy(mask.astype(np.float32))
if use_cuda:
    mask = mask.cuda()
maskedLinear2 = MaskedLinear(infeatures, outfeatures, mask)
activate2 = nn.Sigmoid()
sigmoid = nn.Sigmoid()
if use_cuda:
    maskedLinear2.cuda()
    activate2.cuda()
    sigmoid.cuda()

print("=====Autoencoding layer 2=======")
optimizer = optim.Adam(maskedLinear2.parameters(), lr=args.lr)
criterion = nn.BCELoss()
for epoch in range(args.epochs):
    for batch_idx, (inputs, _) in enumerate(trainloader):
        inputs = inputs.view(inputs.size(0), -1)
        if use_cuda:
                inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs = Variable(inputs)
        inputs = activate1(maskedLinear1(inputs))
        inputs = Variable(inputs.data, requires_grad=False)
        h = activate2(maskedLinear2(inputs))
        outputs = sigmoid(maskedLinear2.decode(h))
        recon_loss = criterion(outputs, inputs)
        recon_loss.backward()
        optimizer.step()

    print("#Epoch %3d: Reconstruct Loss: %.3f" % (epoch, recon_loss.data[0]))

print("=========Classify============")
fc = nn.Linear(outfeatures, num_classes)
if use_cuda:
    fc.cuda()
params = [{'params': maskedLinear1.parameters()},
            {'params': maskedLinear2.parameters()}]
optimizer = optim.Adam(params, lr=args.lr)
criterion = nn.CrossEntropyLoss()
best_acc = 0  # best test accuracy
def test(epoch, dataloader):
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
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

    print("#Epoch %3d: Test Loss: %.3f | Acc: %.3f%%" % (epoch, test_loss/(batch_idx+1),
                                    100.*correct/total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'maskedLinear1': maskedLinear1.state_dict(),
            'maskedLinear2': maskedLinear2.state_dict(),
            'fc': fc.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt-'+args.name+'.t7')
        best_acc = acc

def train(epoch):
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.view(inputs.size(0), -1)
        if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        h1 = activate1(maskedLinear1(inputs))
        h2 = activate2(maskedLinear2(h1))
        outputs = fc(h2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()


    print("#Epoch %3d: Train Loss: %.3f | Acc: %.3f%%" % (epoch, train_loss/(batch_idx+1),
                                    100.*correct/total))

for epoch in range(100):
    train(epoch)
    test(epoch, testloader)

print("Done.")
