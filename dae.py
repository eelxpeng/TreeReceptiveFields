import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import torch.nn.init as init

import numpy as np
import math
from lib.utils import Dataset, readData
from lib.ops import MSELoss

def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise

class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(self.__class__, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.vbias = Parameter(torch.Tensor(in_features))
        
        self.enc_act_func = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.vbias.size(0))
        self.vbias.data.uniform_(-stdv, stdv)

        # init.xavier_uniform(self.weight, gain=np.sqrt(2.0))
        # init.constant(self.bias, 0.1)
        # init.constant(self.vbias, 0.1)

    def encode(self, x, train=True):
        if train:
            self.dropout.train()
        else:
            self.dropout.eval()
        return self.dropout(self.enc_act_func(F.linear(x, self.weight, self.bias)))

    def decode(self, x):
        return F.linear(x, self.weight.t(), self.vbias)

    def fit(self, data_x, valid_x, lr=0.001, batch_size=128, num_epochs=10, corrupt=0.5):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Autoencoding layer=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, betas=(0.9, 0.9))
        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)
        # criterion = nn.MSELoss(size_average=False)
        criterion = nn.MSELoss()
        # criterion = MSELoss()
        trainset = Dataset(data_x, data_x)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        validset = Dataset(valid_x, valid_x)
        validloader = torch.utils.data.DataLoader(
            validset, batch_size=1000, shuffle=False, num_workers=2)

        # validate
        total_loss = 0.0
        total_num = 0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            hidden = self.encode(inputs)
            outputs = self.decode(hidden)

            valid_recon_loss = criterion(outputs, inputs)
            # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
            # total_num += inputs.size()[0]

        # valid_loss = total_loss / total_num
        print("#Epoch 0: Valid Reconstruct Loss: %.3f" % (valid_recon_loss.data[0]))

        for epoch in range(num_epochs):
            # train 1 epoch
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                inputs_corr = masking_noise(inputs, corrupt)
                if use_cuda:
                    inputs = inputs.cuda()
                    inputs_corr = inputs_corr.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                inputs_corr = Variable(inputs_corr)

                hidden = self.encode(inputs_corr)
                outputs = self.decode(hidden)
                recon_loss = criterion(outputs, inputs)
                recon_loss.backward()
                optimizer.step()
                # print("    #Iter %3d: Reconstruct Loss: %.3f" % (
                #     batch_idx, recon_loss.data[0]))

            # validate
            total_loss = 0.0
            total_num = 0
            for batch_idx, (inputs, _) in enumerate(validloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                hidden = self.encode(inputs, train=False)
                outputs = self.decode(hidden)

                valid_recon_loss = criterion(outputs, inputs)
                # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
                # total_num += inputs.size()[0]

            # valid_loss = total_loss / total_num
            print("#Epoch %3d: Reconstruct Loss: %.3f, Valid Reconstruct Loss: %.3f" % (
                epoch, recon_loss.data[0], valid_recon_loss.data[0]))

if __name__ == "__main__":
    # from lib.Tox21_Data import read
    # x_tr_t, y_tr_t, x_valid_t, y_valid_t, x_te_t, y_te_t = read("./dataset/tox21/", target=0)

    label_name = ['World', 'Sports', 'Business', 'Sci/Tech']
    training_num, valid_num, test_num, vocab_size = 110000, 10000, 7600, 10000
    training_file = 'dataset/agnews_training_110K_10K-TFIDF-words.txt'
    valid_file = 'dataset/agnews_valid_10K_10K-TFIDF-words.txt'
    test_file = 'dataset/agnews_test_7600_10K-TFIDF-words.txt'

    randgen = np.random.RandomState(13)
    trainX, trainY = readData(training_file, training_num, vocab_size, randgen)
    validX, validY = readData(valid_file, valid_num, vocab_size)
    testX, testY = readData(test_file, test_num, vocab_size)

    in_features = trainX.size()[1]
    out_features = 500
    dae = DenoisingAutoencoder(in_features, out_features)
    dae.fit(trainX, validX, lr=1e-4, num_epochs=10)
