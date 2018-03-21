import numpy as np
import sys
import pickle
import scipy.io

from lib.bayesnet.discretize import discretize
from lib.bayesnet.chow_liu import chow_liu
from lib.bayesnet.tree_conv import tree_conv
from lib.bayesnet.miislands import miislands
from .utils import TextDataset

# def genMask(data, todiscretize=True, bins=5, kernel_size=1, stride=1):
#     """
#     Input: 
#     *data* : 2-d numpy array, nxd
#         The data from which we will learn. It should be
#         the entire dataset.
#         if continuous, should be discretize
#     Return: masks: a L elements list
#                     Each element is dxh 0-1 matrix
#     """
#     n, d = data.shape
#     if todiscretize:
#         print("Discretizing data into %d bins for each of %d dims" % (bins, d))
#         bins = [bins]*d
#         data = discretize(data, bins=bins, verbose=True)
#     mst = chow_liu(data)
#     with open("mst.pkl", "wb") as f:
#         pickle.dump(mst, f)
#     print("===> Running tree_conv to generate neiborhoods...")
#     islands = tree_conv(mst, 0, kernel_size=kernel_size, stride=stride)

#     out_dim = len(islands)
#     mask = np.zeros((d, out_dim))
#     hidx = 0
#     for h in islands:
#         mask[islands[h], hidx] = 1
#         hidx += 1
#     print("===> Mask generated.")
#     return mask

def genMask(data, thresh=0, kernel_size=1, stride=1):
    """
    Input: 
    *data* : 2-d numpy array, nxd
        The data from which we will learn. It should be
        the entire dataset.
        if continuous, should be discretize
    Return: masks: a L elements list
                    Each element is dxh 0-1 matrix
    """
    n, d = data.shape
    if not isinstance(data, TextDataset):
        data = (data>thresh).astype(np.int8)
    mst = chow_liu(data)
    with open("mst.pkl", "wb") as f:
        pickle.dump(mst, f)
    print("===> Running tree_conv to generate neiborhoods...")
    islands = tree_conv(mst, 0, kernel_size=kernel_size, stride=stride)

    out_dim = len(islands)
    mask = np.zeros((d, out_dim))
    hidx = 0
    for h in islands:
        mask[islands[h], hidx] = 1
        hidx += 1
    print("===> Mask generated.")
    return mask

def genMaskMIIslands(data, thresh=0, kernel_size=1, stride=1):
    """
    Input: 
    *data* : 2-d numpy array, nxd
        The data from which we will learn. It should be
        the entire dataset.
        if continuous, should be discretize
    Return: masks: a L elements list
                    Each element is dxh 0-1 matrix
    """
    n, d = data.shape
    data = (data>thresh).astype(np.int8)
    islands = miislands(data, island_size=kernel_size, stride=stride)
    out_dim = len(islands)
    mask = np.zeros((d, out_dim))
    hidx = 0
    for h in islands:
        mask[islands[h], hidx] = 1
        hidx += 1
    print("===> Mask generated.")
    return mask

if __name__ == "__main__":
    num_obs = 1152
    vars = []
    for i in range(num_obs):
        vars.append("v%d" % i)
    masks = loadSparseNet("corrList.csv", vars, 0.1)
    print("Max number of layers=%d" % len(masks))
    for i in range(len(masks)):
        print("Layer %d: (%dx%d)" % (i, masks[i].shape[0], masks[i].shape[1]))
