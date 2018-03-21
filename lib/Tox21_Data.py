author='chen zhourong'
# make sure  numpy, scipy, pandas, sklearn are installed, otherwise run
# pip install numpy scipy pandas scikit-learn
import numpy as np
import pandas as pd
from scipy import io
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from torch import FloatTensor, LongTensor
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def read(path, target=0, valid_num=1000, randseed=113):
	r"""
	target:		task index, range [0-11]
	valid_num:	number of validation datacases
	randseed:	random seed for permutating training data

	return:	torch.FloatTensor(trainX),	torch.LongTensor(trainY),
			torch.FloatTensor(validX),	torch.LongTensor(validY),
			torch.FloatTensor(testX),	torch.LongTensor(testY)
	"""

	randgen = np.random.RandomState(randseed)

	# load data
	y_tr = pd.read_csv(path+'tox21_labels_train.csv.gz', index_col=0, compression="gzip")
	y_te = pd.read_csv(path+'tox21_labels_test.csv.gz', index_col=0, compression="gzip")
	x_tr_dense = pd.read_csv(path+'tox21_dense_train.csv.gz', index_col=0, compression="gzip").values
	x_te_dense = pd.read_csv(path+'tox21_dense_test.csv.gz', index_col=0, compression="gzip").values
	x_tr_sparse = io.mmread(path+'tox21_sparse_train.mtx').tocsc()
	x_te_sparse = io.mmread(path+'tox21_sparse_test.mtx').tocsc()

	# filter out very sparse features
	filter_threshold = 0.05
	print('Filtering sparse features with threshold: %f' % filter_threshold)
	sparse_col_idx = ((x_tr_sparse > 0).mean(0) > filter_threshold).A.ravel()
	x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].A])
	x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].A])

	# filter out datacases without label
	print('Target: %d, %s' % (target, y_tr.columns[target]))
	target = y_tr.columns[target]
	print('Total training data before filtering out datacases without label: %d' % x_tr.shape[0])
	print('Total test data before filtering out datacases without label: %d' % x_te.shape[0])
	rows_tr = np.isfinite(y_tr[target]).values
	rows_te = np.isfinite(y_te[target]).values
	x_tr, y_tr = x_tr[rows_tr], y_tr[target][rows_tr].values
	x_te, y_te = x_te[rows_te], y_te[target][rows_te].values

	# permutate training data
	rand_idx = randgen.permutation(x_tr.shape[0])
	x_tr, y_tr = x_tr[rand_idx], y_tr[rand_idx]

	# scale data
	train_mean = x_tr[valid_num:].mean(0)
	train_std = x_tr[valid_num:].std(0)
	train_std[train_std==0.0] = 1.0
	x_tr = (x_tr - train_mean)/train_std
	x_te = (x_te - train_mean)/train_std

	# build tensor
	x_tr_t, y_tr_t = FloatTensor(x_tr[valid_num:].astype(np.float32)), LongTensor(y_tr[valid_num:].astype(np.int64))
	x_valid_t, y_valid_t = FloatTensor(x_tr[:valid_num].astype(np.float32)), LongTensor(y_tr[:valid_num].astype(np.int64))
	x_te_t, y_te_t = FloatTensor(x_te.astype(np.float32)), LongTensor(y_te.astype(np.int64))
	print('Final training data size: ' + str(x_tr_t.size()))
	print('Final validation data size: ' + str(x_valid_t.size()))
	print('Final test data size: ' + str(x_te_t.size()))
	return x_tr_t, y_tr_t, x_valid_t, y_valid_t, x_te_t, y_te_t

