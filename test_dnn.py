import argparse
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

from lib.utils import readData, Dataset

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
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

def fit(net, trainX, trainY, validX, validY, testX, testY,
        batch_size=256, lr=0.01, epochs=10, name="dnn"):
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

        print("\nBest Valid ACC=%.3f, test ACC=%.3f" % (best_valid_acc, best_test_acc))


if args.dataset=="agnews":
    label_name = ['World', 'Sports', 'Business', 'Sci/Tech']
    training_num, valid_num, test_num, vocab_size = 110000, 10000, 7600, 10000
    training_file = 'dataset/agnews_training_110K_10K-TFIDF-words.txt'
    valid_file = 'dataset/agnews_valid_10K_10K-TFIDF-words.txt'
    test_file = 'dataset/agnews_test_7600_10K-TFIDF-words.txt'

elif args.dataset=="dbpedia":
    label_name = ['Company','EducationalInstitution','Artist','Athlete','OfficeHolder','MeanOfTransportation','Building','NaturalPlace','Village','Animal','Plant','Album','Film','WrittenWork']
    training_num, valid_num, test_num, vocab_size = 549990, 10010, 70000, 10000
    training_file = 'dataset/dbpedia_training_549990_10K-frequent-words.txt'
    valid_file = 'dataset/dbpedia_valid_10010_10K-frequent-words.txt'
    test_file = 'dataset/dbpedia_test_70K_10K-frequent-words.txt'

elif args.dataset=="sogounews":
    label_name = ['sports','finance','entertainment','automobile','technology']
    training_num, valid_num, test_num, vocab_size = 440000, 10000, 60000, 10000
    training_file = 'dataset/sogounews_training_440K_10K-frequent-words.txt'
    valid_file = 'dataset/sogounews_valid_10K_10K-frequent-words.txt'
    test_file = 'dataset/sogounews_test_60K_10K-frequent-words.txt'

elif args.dataset=="yelp":
    label_name = ['1','2','3','4','5']
    training_num, valid_num, test_num, vocab_size = 640000, 10000, 50000, 10000
    training_file = 'dataset/yelpfull_training_640K_10K-frequent-words.txt'
    valid_file = 'dataset/yelpfull_valid_10K_10K-frequent-words.txt'
    test_file = 'dataset/yelpfull_test_50K_10K-frequent-words.txt' 

elif args.dataset=="yahoo":
    label_name = ["Society & Culture","Science & Mathematics","Health","Education & Reference","Computers & Internet","Sports","Business & Finance","Entertainment & Music","Family & Relationships","Politics & Government"]
    training_num, valid_num, test_num, vocab_size = 1390000, 10000, 60000, 10000
    training_file = 'dataset/yahoo_training_1.39M_10K-frequent-words.txt'
    valid_file = 'dataset/yahoo_valid_10K_10K-frequent-words.txt'
    test_file = 'dataset/yahoo_test_60K_10K-frequent-words.txt'

print("Running Dataset %s with [lr=%f, epochs=%d, batch_size=%d]"
        % (args.dataset, args.lr, args.epochs, args.batch_size))

num_classes = len(label_name)
randgen = np.random.RandomState(13)
trainX, trainY = readData(training_file, training_num, vocab_size, randgen)
validX, validY = readData(valid_file, valid_num, vocab_size)
testX, testY = readData(test_file, test_num, vocab_size)

input_dim = trainX.size()[1]
dims = [input_dim, 1100, 550, 275, num_classes]

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

net = FNN(dims)
fit(net, trainX, trainY, validX, validY, testX, testY, batch_size=args.batch_size,
    lr=args.lr, epochs=args.epochs, name=args.name)
print("Done.")
