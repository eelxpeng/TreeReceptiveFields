"""
*************
Chow-Liu Tree
*************

Calculate the KL divergence (i.e. run
mi_test) between every pair of nodes,
then select the maximum spanning tree from that
connected graph. This is the Chow-Liu tree.

"""

__author__ = """Xiaopeng LI <xlibo@connect.ust.hk>"""

from .utils import mutual_information, mutual_information_binary
# from .utils import mutual_information_fast
from .bayesnet import BayesNet
import operator
import numpy as np
import scipy.io

def chow_liu(data, directed=False, binary=True):
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
    value_dict = dict(zip(range(data.shape[1]),
        [list(np.unique(col)) for col in data.T]))

    n_rv = data.shape[1]
    print("===> Number of variables: %d" % n_rv)

    print("===> Computing pairwise mutual_information...")
    
    if binary:
        edge_list = mutual_information_binary(data)
    else:
        edge_list = []
        for i in range(n_rv):
            for j in range(i+1, n_rv):
                edge_list.append((i,j,mutual_information(data[:,(i,j)])))
            if (i+1)%10==0:
                print("MI computed for %d variables..." % (i+1))
    print("===> Number of considerred edges: %d" % len(edge_list))
    print("===> Running Kruskal's algorithm...")
    mst = mst_kruskal(edge_list, n_rv)
    print("===> Done. Maximum Spanning Tree Generated.")
    # need to convert undirected graph into directed
    # if BayesNet is needed.
    # here set vertex 0 as root
    if directed:
        undirectedToDirected(mst, 0)
    
    return mst

def mst_kruskal(edge_list, n_rv):
    """
    Kruskal's algorithm
    by Xiaopeng LI
    """
    edge_list.sort(key=operator.itemgetter(2), reverse=True) # sort by weight
    mst = dict((rv, []) for rv in range(n_rv))
    # make set for each vertex
    components = dict()
    for i in range(n_rv):
        components[i] = [i]

    for i,j,w in edge_list:
        # since undirected, i->j and j-> is the same
        # and in edge_list, there are only i->j
        if(components[i]!=components[j]):
            mst[i].append(j)
            mst[j].append(i)
            if((len(components[i])+len(components[j]))==n_rv):
                break
            # merge connected component
            components[i].extend(components[j])
            for c in components[j]:
                components[c] = components[i]

    return mst

def undirectedToDirected(mst, root):
    """
    convert undirected spanning tree to directed spanning tree
    by Xiaopeng LI
    Arguments
    ----------
    mst: dict, key is the vertex, value is adj list
    root: the vertex as root

    Notes
    -----
        Implemented through dfs
    """
    if len(mst[root])==0:
        return
    for v in mst[root]:
        mst[v].remove(root)
        undirectedToDirected(mst, v)

if __name__ == "__main__":
    edge_list = [(0, 1, -4), (0, 7, -8), 
                (1, 2, -8), (1, 7, -11),
                (2, 3, -7), (2, 5, -4), (2, 8, -2),
                (3, 4, -9), (3, 5, -14),
                (4, 5, -10),
                (5, 6, -2),
                (6, 7, -1), (6, 8, -6),
                (7, 8, -7)]
    mst = mst_kruskal(edge_list, 9)
    for v in mst:
        print(v, mst[v])
