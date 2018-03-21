import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import gensim
import argparse
import sys
# from lib.utils import readData
# from ops import *

def mutual_information(data):
    # mutual information for binary data
    hist,_ = np.histogramdd(data, bins=2) # frequency counts

    Pxy = hist / hist.sum()# joint probability distribution over X,Y,Z
    Px = np.sum(Pxy, axis = 1) # P(X,Z)
    Py = np.sum(Pxy, axis = 0) # P(Y,Z) 

    PxPy = np.outer(Px,Py)
    Pxy += 1e-7
    PxPy += 1e-7
    MI = np.sum(Pxy * np.log(Pxy / (PxPy)))
    return round(MI,4)

def cosine_sim(data1, data2):
    # assuming binary
    # return np.sum(data1*data2)/((np.sum(data1)+1e-7)*(np.sum(data2)+1e-7))
    num = np.dot(data1.T, data2)
    den = np.outer(np.sum(data1, axis=0), np.sum(data2, axis=0))+1e-7
    return num/den

def compactness(topic, word2vec):
    M = len(topic)
    sum_co = 0
    for i in range(1,M):
        for j in range(i):
            wi = topic[i]
            wj = topic[j]
            if i in word2vec.vocab and wj in word2vec.vocab:
                sum_co += word2vec.similarity(wi, wj)
    ave_co = 2.0/(M*(M-1)) * sum_co
    return ave_co

def interpret(net, data, words, top=10, index=None, batch_size=1000):
    use_cuda = torch.cuda.is_available()
    net.eval()
    if use_cuda:
        net.cuda()

    n = data.size()[0]
    hiddens = []
    num_batches = int(math.ceil(1.0*n/batch_size))

    print("projecting data...")
    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    if index is not None:
        net[index].register_forward_hook(hook_feature)
    else:
        net.register_forward_hook(hook_feature)

    for i in range(num_batches):
        begin = i*batch_size
        end = min((i+1)*batch_size, n)
        batchX = data[begin: end]
        del features_blobs[:]
        if use_cuda:
            batchX = batchX.cuda()
        batchX = Variable(batchX)
        output = net(batchX)
        hiddens.append(np.copy(features_blobs[0]))

    hiddens = np.concatenate(hiddens)
    print("hiddens.shape=", hiddens.shape)
    threshold = 0.0
    hiddens = (hiddens>threshold).astype(np.float32)

    # compute mutual information for every pair of h and w
    print("computing mutual information between hidden and words...")
    num_hidden = hiddens.shape[1]
    num_obs = data.shape[1]
    mis = np.zeros((num_obs, num_hidden))
    # for j in range(num_hidden):
    #     for i in range(num_obs):
    #         # pair = np.concatenate((data[:, [i]].cpu().numpy(), hiddens[:, [j]]), axis=1)
    #         # mis[i, j] = mutual_information(pair)
    #         mis[i,j] = cosine_sim(data[:, i].cpu().numpy(), hiddens[:, j])

    #     progress = 1.0*j/num_hidden*100
    #     sys.stdout.write('\r[%-10s] %0.2f%%' % ('#' * int(progress/10), progress))
    #     sys.stdout.flush()
    mis[:] = cosine_sim(data.cpu().numpy(), hiddens)

    hw = np.argsort(mis, axis=0)[::-1, :]

    # print top k words for each hidden units
    # for i in range(num_hidden):
    #     w = [words[j] for j in hw[:top, i]]
    #     print("#h%d: %s" % (i, " ".join(w)))

    # compute compactness score for each hidden unit
    print("computing compactness score...")
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)  
    sum_compact = np.array([0.0]*num_hidden)
    for i in range(num_hidden):
        w = [words[j] for j in hw[:top, i]]
        sum_compact[i] = compactness(w, model)

        progress = 1.0*i/num_hidden*100
        sys.stdout.write('\r[%-10s] %0.2f%%' % ('#' * int(progress/10), progress))
        sys.stdout.flush()

    ind = np.argsort(sum_compact)[::-1]
    for i in range(100):
        w = [words[j] for j in hw[:top, ind[i]]]
        print("#h%d [%f]: %s" % (i, sum_compact[ind[i]], " ".join(w)))

    ave_compactness = np.mean(sum_compact)
    print("Average compactness score=%f" % ave_compactness)

def load_vocab(filename):
    with open(filename) as fid:
        for line in fid:
            vocab = line.strip().split(",")
            return vocab

def read_max(filename, data_num, vocab_size):
    dataX = torch.FloatTensor(1, vocab_size) *0
    data_max = torch.FloatTensor(1, vocab_size) *0
    infile = open(filename)
    count = 0
    for line in infile:
        line = line.strip('\n').split(',')
        # dataY[ idx[count] ] = int(line[0])

        entry_list = [[int(listed_pair[0]), int(listed_pair[1])] for listed_pair in [pair.split(':') for pair in line[1:]]]
        entry_tensor = torch.LongTensor(entry_list)
        if len(entry_list)!=0:
            dataX[ 0 ][entry_tensor[:,0]] = entry_tensor[:,1].type(torch.FloatTensor)
        data_max[:] = torch.max(data_max, dataX)
        count += 1
        if count%10000==0:
            print("%d data read." % count)
    infile.close()
    assert count == data_num, (count, data_num)
    print('Read %d\t datacases\t Done!\n' % count)
    return data_max

def readData(filename, data_num, vocab_size, randgen=None):
    dataX = torch.FloatTensor(data_num, vocab_size) *0
    dataY = torch.LongTensor(data_num) *0
    if randgen != None:
        print('Reading data with permutation from %s\n' % filename)
        idx = randgen.permutation(data_num)
    else:
        print('Reading data without permutation from %s\n' % filename)
        idx = range(data_num)

    infile = open(filename)
    count = 0
    for line in infile:
        line = line.strip('\n').split(',')
        dataY[ idx[count] ] = int(line[0])

        entry_list = [[int(listed_pair[0]), int(listed_pair[1])] for listed_pair in [pair.split(':') for pair in line[1:]]]
        entry_tensor = torch.LongTensor(entry_list)
        if len(entry_list)!=0:
            dataX[ idx[count] ][entry_tensor[:,0]] = entry_tensor[:,1].type(torch.FloatTensor)
        count += 1
        if count%10000==0:
            print("%d data read." % count)
    infile.close()
    assert count == data_num, (count, data_num)
    print('Read %d\t datacases\t Done!\n' % count)

    return dataX, dataY

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='DNN parameters')
    parser.add_argument('--model', type=str, default="checkpoint/ckpt-prune-agnews-l3-rec-2048.t7",
                        help='model path')
    parser.add_argument('--dataset', type=str, default="agnews",
                        help='dataset')
    parser.add_argument('--vocab', type=str, default="dict", 
                        help='vocab path')
    parser.add_argument('--top', type=int, default=10, 
                        help='use top k words for each hidden unit')
    args = parser.parse_args()
    transform = False
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
        transform = True

    num_classes = len(label_name)
    randgen = np.random.RandomState(13)
    # trainX, trainY = readData(training_file, training_num, vocab_size, randgen)
    # validX, validY = readData(valid_file, valid_num, vocab_size)
    testX, testY = readData(test_file, test_num, vocab_size)

    if transform:
        # preprocess, normalize each dimension to be [0, 1] for cross-entropy loss
        # train_max = torch.max(trainX, dim=0, keepdim=True)[0]
        # valid_max = torch.max(validX, dim=0, keepdim=True)[0]
        # test_max = torch.max(testX, dim=0, keepdim=True)[0]
        train_max = read_max(training_file, training_num, vocab_size)
        valid_max = read_max(valid_file, valid_num, vocab_size)
        test_max = read_max(test_file, test_num, vocab_size)
        print(train_max.size())
        print(valid_max.size())
        print(test_max.size())
        x_max = torch.max(torch.cat((train_max, valid_max, test_max), 0), dim=0, keepdim=True)[0]
        # trainX.div_(x_max)
        # validX.div_(x_max)
        testX.div_(x_max)

    blob = "net"
    index = -2

    state = torch.load(args.model, map_location=lambda storage, loc: storage)
    print("Acc: ", state["acc"])
    net = state["net"]

    print(net.__class__)
    vocab = load_vocab(args.vocab)
    interpret(net, testX, vocab, top=args.top, index=index)