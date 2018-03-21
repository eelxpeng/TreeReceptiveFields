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
parser.add_argument('--sparsity', type=float, default=0.1, 
                    help='pruning sparsity')
parser.add_argument("--globalprune", help="use globalprune or not",
                    action="store_true")
parser.add_argument('--name', type=str, default="tox-1-512",
                    help='name for this run')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
batchsize = args.batch_size

init_logging("log-"+args.name)

if args.dataset=="tox":
    logging.info("Running Dataset %s Target %d with [num_layer=%d, shape=%d, num-neuron=%d] and [epochs=%d, batch_size=%d]"
        % (args.dataset, args.target, args.num_layer, args.shape, args.num_neuron, args.epochs, args.batch_size))
else:
    logging.info("Running Dataset %s with [num_layer=%d, shape=%d, num-neuron=%d] and [epochs=%d, batch_size=%d]"
        % (args.dataset, args.num_layer, args.shape, args.num_neuron, args.epochs, args.batch_size))
logging.info("Prune sparsity=%.2f, globalprune=%s" % (args.sparsity, "yes" if args.globalprune else "no"))

if args.dataset=="tox":
    label_name = ['0','1']
    from Tox21_Data import read
    trainX, trainY, validX, validY, testX, testY = read("tox21/", target=args.target, valid_num=500)
    training_num, test_num, vocab_size = trainX.size()[0], testX.size()[0], trainX.size()[1]

else:
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

    randgen = np.random.RandomState(13)
    trainX, trainY = readData(training_file, training_num, vocab_size, randgen)
    validX, validY = readData(valid_file, valid_num, vocab_size)
    testX, testY = readData(test_file, test_num, vocab_size)

batchnum = int(np.ceil(trainX.size()[0] / batchsize))
logging.info( 'Average length of training documents: %f\n\n' % (trainX.sum()/float(trainX.size()[0])) )

input_size = trainX.size()[1]
num_classes = len(label_name)
dims = [input_size]
dims += [args.num_neuron]
prev = args.num_neuron
ratio = math.pow(1.0*args.num_neuron/num_classes, 1.0/args.num_layer)
for i in range(1, args.num_layer):
    if args.shape==0:
        dims += [prev]
    else:
        prev = int(prev/ratio)
        dims += [prev]
dims += [num_classes]

# def zero_grad(self, grad_input, grad_output):
#     return grad_input * self.mask

# class MaskedLinear(nn.Module):
#     def __init__(self, in_features, out_features, mask):
#         super(MaskedLinear, self).__init__()
#         self.linear = nn.Linear(in_features, out_features)
#         # weight dim is out_feature x in_feature
#         self.linear.weights *= mask  # to zero it out first
#         self.mask = mask
#         self.handle = self.register_backward_hook(zero_grad)  # to make sure gradients won't propagate

#     def forward(self, x):
#         return self.linear(x)
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

class SparseNN(nn.Module):
    def __init__(self, dims, masks, dropouts):
        super(self.__class__, self).__init__()

        self.sparse = []
        for i in range(len(dims)-2):
            self.sparse.append( MaskedLinear(dims[i], dims[i+1], masks[i]) )
            self.sparse.append( nn.ReLU() )
            self.sparse.append( nn.Dropout(p=dropouts[i]) )
        self.sparse.append( MaskedLinear(dims[-2], dims[-1], masks[-1]) )
        
        self.sparse = nn.Sequential(*self.sparse)

    def forward(self, x):
        return self.sparse(x)

    def copyParameterFrom(self, fnn):
        num_layers = len(fnn.fc)
        assert(num_layers==len(self.sparse))
        for l in range(num_layers):
            if isinstance(fnn.fc[l], nn.Linear):
                self.sparse[l].weight.data.copy_(fnn.fc[l].weight.data)
                self.sparse[l].bias.data.copy_(fnn.fc[l].bias.data)

def prune(fnn, keep_ratio=0.1):
    masks =[]
    dropouts = []
    for l in range(len(fnn.fc)):
        if isinstance(fnn.fc[l], nn.Linear):
            weight = np.abs(fnn.fc[l].weight.data.cpu().numpy())
            num_links = weight.shape[0]*weight.shape[1]
            num_left = int(weight.shape[0]*weight.shape[1]*keep_ratio)
            threshold = np.sort(weight.reshape(-1))[::-1][num_left-1]
            mask = (weight>=threshold).astype(np.float32)
            num_links_mask = np.sum(mask)
            logging.info("Layer #%d (%d,%d): %d->%d" % (l, weight.shape[1],
                weight.shape[0], num_links, num_links_mask))
            mask = torch.from_numpy(mask)
            masks.append(mask)
            dropouts.append(dropout*math.sqrt(num_links_mask/num_links))

    return (masks, dropouts)

def globalprune(fnn, keep_ratio=0.1):
    weights = []
    for l in range(len(fnn.fc)):
        if isinstance(fnn.fc[l], nn.Linear):
            weight = np.abs(fnn.fc[l].weight.data.cpu().numpy()).reshape(-1)
            weights.append(weight)
    weights = np.concatenate(weights)
    num_links = len(weights)
    num_left = int(num_links*keep_ratio)
    threshold = np.sort(weights)[::-1][num_left-1]
    masks =[]
    dropouts = []
    for l in range(len(fnn.fc)):
        if isinstance(fnn.fc[l], nn.Linear):
            weight = np.abs(fnn.fc[l].weight.data.cpu().numpy())
            num_links = weight.shape[0]*weight.shape[1]
            mask = (weight>=threshold).astype(np.float32)
            num_links_mask = np.sum(mask)
            logging.info("Layer #%d (%d,%d): %d->%d" % (l, weight.shape[1],
                weight.shape[0], num_links, num_links_mask))
            mask = torch.from_numpy(mask)
            masks.append(mask)
            dropouts.append(dropout*math.sqrt(num_links_mask/num_links))

    return (masks, dropouts)


def train(net):
    global validX, validY
    if use_cuda:
        net = net.cuda()
    logging.info(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam( filter(lambda p: p.requires_grad, net.parameters()))
    scheduler = LambdaLR( optimizer, lr_lambda=lambda epoch:1.0/np.sqrt(epoch+1) )
    #optimizer = optim.SGD( filter(lambda p: p.requires_grad, bridgeNet.parameters()), lr=0.01, momentum=0.5, nesterov=True )

    best_loss = 1000000
    best_acy = 0.0
    best_loss_acy = 0.0
    best_auc = 0.0
    best_loss_auc = 0.0
    loss_history = []
    patience = 40
    early_stop = False
    for epoch in range(args.epochs):
        running_loss = 0.0
        scheduler.step()
        for batch in range(batchnum):
            begin = batch * batchsize
            if batch == batchnum-1:
                end = trainX.size()[0]
            else:
                end = begin + batchsize

            batchX = trainX[begin: end]
            batchY = trainY[begin: end]
            if use_cuda:
                batchX = batchX.cuda()
                batchY = batchY.cuda()
            batchX = Variable(batchX)
            batchY = Variable(batchY)

            net.train(mode=True)
            optimizer.zero_grad()

            #forward, backward, update
            output = net(batchX)
            loss = criterion(output, batchY)
            loss.backward()
            optimizer.step()

            # logging.info statistics
            running_loss += loss.data[0]
            if batch % 10 == 9:
                logging.info('[Epoch #%d, Iter #%5d] training batch loss: %.3f' % (epoch+1, batch+1, running_loss / 10))
                running_loss = 0.0

                net.train(mode=False)

                if use_cuda:
                    validX = validX.cuda()
                    validY = validY.cuda()
                output = net(Variable(validX, volatile=True))
                assert( output.size()[1] == len(label_name) ) , (output.size()[1], len(label_name))
                target = Variable(validY, volatile=True)
                loss = criterion(output, target)
                loss_history.append(loss.data[0])
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == target.data).sum()
                if args.dataset=="tox":
                    auc_score = roc_auc_score(validY.cpu().numpy(), F.softmax(output)[:,1].data.cpu().numpy())

                logging.info('Validation loss: %.3f' % (loss.data[0]))
                if loss.data[0] < best_loss:
                    best_loss = loss.data[0]
                    best_loss_acy = 100.0 * correct / predicted.size()[0]
                    if args.dataset=="tox":
                        best_loss_auc = auc_score

                logging.info('Best validation loss: %.3f' % best_loss)
                logging.info('Accuracy at best validation loss: %.2f %%' % best_loss_acy)
                if args.dataset=="tox":
                    logging.info('AUC at best validation loss: %.4f' % best_loss_auc)

                logging.info('Accuracy of the network on validation docs: %.2f %%' % (
                    100.0 * correct / predicted.size()[0]))
                if args.dataset=="tox":
                    logging.info('Auc score on validation data: %.4f' % auc_score)

                if 100.0 * correct / predicted.size()[0] > best_acy:
                    best_acy = 100.0 * correct / predicted.size()[0]
                logging.info('Best validation accuracy: %.2f %%' % best_acy)
                if args.dataset=="tox":
                    if auc_score > best_auc:
                        best_auc = auc_score
                    logging.info('Best validation auc score: %.4f' % best_auc)

                if loss.data[0] <= np.array(loss_history).min():
                    bad_count = 0
                    best_model = copy.deepcopy(net)
                if len(loss_history)>patience and loss.data[0] >= np.array(loss_history)[:-patience].min():
                    bad_count += 1
                    if bad_count > patience:
                        logging.info('Early stop!')
                        early_stop = True
                        break

        if early_stop:
            break

    return best_model

def test(net):
    global testX, testY
    net.train(mode=False)
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        testX = testX.cuda()
        testY = testY.cuda()
    output = net(Variable(testX, volatile=True))
    assert( output.size()[1] == len(label_name) ) , (output.size()[1], len(label_name))
    target = Variable(testY, volatile=True)
    loss = criterion(output, target)
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target.data).sum()
    if args.dataset=="tox":
        auc_score = roc_auc_score(testY.cpu().numpy(), F.softmax(output)[:,1].data.cpu().numpy())
    logging.info('\n\nFinal test loss: %.3f' % loss.data[0])
    logging.info('Final test accuracy: %.2f %%' % (100.0 * correct / predicted.size()[0]))
    if args.dataset=="tox":
        logging.info('Final test AUC score: %.4f' % auc_score)

logging.info("===========Training FNN=======================")
fnn = FNN(dims)
fnn = train(fnn)
test(fnn)

logging.info("===========Training Prune=======================")
if args.globalprune:
    masks, dropouts = globalprune(fnn, keep_ratio=args.sparsity)
else:
    masks, dropouts = prune(fnn, keep_ratio=args.sparsity)
sparse = SparseNN(dims, masks, dropouts)
sparse.copyParameterFrom(fnn)
sparse = train(sparse)
test(sparse)
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
save_path='./checkpoint/ckpt-'+args.name+'.t7'
torch.save(sparse, save_path)
