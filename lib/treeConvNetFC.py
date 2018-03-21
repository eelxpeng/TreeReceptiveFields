import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import scipy.io
import os
import sys
import pickle
from sklearn.metrics import roc_auc_score

from lib.genMask import genMask
from lib.ops import MaskedLinear
from lib.Tox21_Data import Dataset, read
from lib.utils import readData
from lib.makeLayer import makeLayerMaskedDAEFC

class TreeConvNetFC:
    def __init__(self, name="treeconv"):
        self.name = name
        self.net = None

    def learn_structure(self, trainX, validX, num_class, kernel_stride, fcwidths, corrupt=0.5,
        lr=1e-3, batch_size=256, epochs=10, loss_type="mse"):
        layers = []
        tr_x = trainX
        va_x = validX
        assert(len(kernel_stride)==len(fcwidths))
        num_layers = len(kernel_stride)
        for l in range(num_layers):
            k, s = kernel_stride[l]
            fw = fcwidths[l]
            if l==0:
                layer = makeLayerMaskedDAEFC(tr_x, va_x, corrupt=corrupt, kernel_size=k, stride=s, fcwidth=fw,
                    lr=lr, batch_size=batch_size, epochs=epochs, loss_type=loss_type)    
            else:
                layer = makeLayerMaskedDAEFC(tr_x, va_x, corrupt=corrupt, kernel_size=k, stride=s, fcwidth=fw,
                    lr=lr, batch_size=batch_size, epochs=epochs)
            layer.eval()
            tr_x = layer.encodeBatch(tr_x)
            va_x = layer.encodeBatch(va_x)
            layers.append(layer)
            
        last_hidden = layers[-1].weight.data.size()[0] + layers[-1].weight_fc.data.size()[0]
        layers.append(nn.Linear(last_hidden, num_class))
        self.net = nn.Sequential(*layers)

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(self.net, './checkpoint/ckpt-'+self.name+'-structure.t7')

    def fit(self, trainX, trainY, validX, validY, testX, testY,
        batch_size=256, lr=0.01, epochs=10, anneal="lambda", weight_decay=0):
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

        print(self.net)
        if use_cuda:
            self.net.cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr,
            weight_decay=weight_decay)
        if anneal=="lambda":
            scheduler = LambdaLR( optimizer, lr_lambda=lambda epoch:1.0/np.sqrt(epoch+1) )
        elif anneal=="step":
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        best_valid_acc = 0  # best test accuracy
        best_test_acc = 0

        def test(net, epoch, dataloader):
            net.eval()
            test_loss = 0
            correct = 0
            total = 0

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

            acc = 100.*correct/total
            print("#Epoch %3d: Test Loss: %.3f | Acc: %.3f%%" % (epoch, test_loss/(batch_idx+1), 
                acc))
            
            return acc

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

            print("#Epoch %3d: Train Loss: %.3f | Acc: %.3f%%" % (epoch, train_loss/(batch_idx+1),
                                            100.*correct/total))

        for epoch in range(epochs):
            scheduler.step()
            train(self.net, epoch)
            accValid = test(self.net, epoch, validloader)
            accTest = test(self.net, epoch, testloader)

            if accValid > best_valid_acc:
                print('Saving..')
                state = {
                    'net': self.net,
                    'acc': accValid,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt-'+self.name+'.t7')
                best_valid_acc = accValid
                best_test_acc = accTest

        print("\nBest Valid ACC=%.3f, test ACC=%.3f" % (best_valid_acc, best_test_acc))
        return (best_valid_acc, best_test_acc)
