"""
*************
MIIslands based on Chow Liu Tree
Center of islands are generated by chow liu tree
Other members of islands are most correlated variables
*************
"""

__author__ = """Xiaopeng LI <xlibo@connect.ust.hk>"""

import numpy as np
from collections import deque
from .chow_liu import mst_kruskal
from .tree_conv import tree_conv


def mutual_information_binary(data):
    """
    data: binary format
    return mi_matrix
    """
    num, n_rv = data.shape
    frequency = np.zeros((n_rv, n_rv), dtype=np.int32)
    print("Computing single and joint counts...")
    for d in data:
        index = np.nonzero(d)[0]
        for i in index:
            frequency[i, index] += 1

    mi_matrix = np.zeros((n_rv, n_rv))
    Pxy = np.zeros((2, 2), dtype=np.float32)
    Px = np.zeros(2, dtype=np.float32)
    Py = np.zeros(2, dtype=np.float32)
    for i in range(n_rv):
        for j in range(i+1, n_rv):
            Px[1] = 1.0*frequency[i, i]/num
            Px[0] = 1-Px[1]
            Py[1] = 1.0*frequency[j, j]/num
            Py[0] = 1-Py[1]
            Pxy[1, 1] = 1.0*frequency[i, j]/num
            Pxy[1, 0] = Px[1] - Pxy[1, 1]
            Pxy[0, 1] = Py[1] - Pxy[1, 1]
            Pxy[0, 0] = 1 - Px[1] - Py[1] + Pxy[1, 1]

            PxPy = np.outer(Px,Py)
            Pxy += 1e-7
            PxPy += 1e-7
            MI = np.sum(Pxy * np.log(Pxy / (PxPy)))

            mi_matrix[i, j] = MI
            mi_matrix[j, i] = MI

        if (i+1)%1000==0:
            print("MI computed for %d variables..." % (i+1))

    return mi_matrix

def miislands(data, island_size=10, stride=2):
    """
    Perform Chow-Liu structure learning algorithm
    over an entire dataset, and return the BN-tree.


    Arguments
    ---------
    *data* : a nested numpy array
        The data from which we will learn. It should be
        the entire dataset.

    Returns
    -------
    *mst* : undirected maximum spanning tree

    Effects
    -------
    None

    Notes: Prim's algorithm is better. Since time complexity is O(VlogV).
            Kruskal's algorithm is O(ElogV). Both space and time inefficient in
            the case of dense link.
    -----

    """

    n_rv = data.shape[1]
    print("===> Number of variables: %d" % n_rv)

    print("===> Computing pairwise mutual_information...")
    
    mi_matrix = mutual_information_binary(data)

    ind_i, ind_j = np.triu_indices(n_rv, 1)
    assert(int(n_rv*(n_rv-1)/2) == len(ind_i))
    edge_list = [None]*int(n_rv*(n_rv-1)/2)
    ind = -1
    for i in range(len(ind_i)):
    	ind += 1
    	edge_list[ind] = (ind_i[i], ind_j[i], mi_matrix[ind_i[i], ind_j[i]])
    print("===> Number of considerred edges: %d" % len(edge_list))
    print("===> Running Kruskal's algorithm...")
    mst = mst_kruskal(edge_list, n_rv)
    del edge_list
    print("===> Done. Maximum Spanning Tree Generated.")

    print("===> Running tree_conv to generate neiborhoods...")
    islands = tree_conv(mst, 0, kernel_size=stride, stride=stride)

    ind_sorted = np.argsort(mi_matrix, axis=1)[:, ::-1]
    for vertex in islands:
    	islands[vertex] = ind_sorted[vertex, :island_size]
    
    return islands


if __name__ == "__main__":
	mst = dict()
	mst[0] = [1,2]
	mst[1] = [0, 3, 4]
	mst[2] = [0,5,6]
	mst[3] = [1,7,8]
	mst[4] = [1,9]
	mst[5] = [2]
	mst[6] = [2]
	mst[7] = [3]
	mst[8] = [3]
	mst[9] = [4]
	islands = tree_conv(mst, 0, kernel_size=2, stride=2)
	for e in islands:
		print(e, islands[e])
