"""
******************************
Mutual Independence
for Structure Learning
******************************
"""
import numpy as np
from scipy import stats
import pdb

def mutual_information_binary_dataset(data):
    """
    data: Dataset
    """
    num, n_rv = data.shape
    frequency = np.zeros((n_rv, n_rv), dtype=np.int32)
    print("Computing single and joint counts...")
    for i in range(num):
        d,_ = data[i]
        d = d.numpy()
        index = np.nonzero(d)[0]
        for i in index:
            frequency[i, index] += 1

    edge_list = [None]*int(n_rv*(n_rv-1)/2)
    ind = -1
    Pxy = np.zeros((2, 2), dtype=np.float32)
    Px = np.zeros(2, dtype=np.float32)
    Py = np.zeros(2, dtype=np.float32)
    print("Computing MI between variables...")
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

            ind += 1
            edge_list[ind] = (i, j, MI)

        if (i+1)%1000==0:
            print("MI computed for %d variables..." % (i+1))

    return edge_list

def mutual_information_binary(data):
    """
    data: binary format
    """
    num, n_rv = data.shape
    frequency = np.zeros((n_rv, n_rv), dtype=np.int32)
    print("Computing single and joint counts...")
    for d in data:
        index = np.nonzero(d)[0]
        for i in index:
            frequency[i, index] += 1

    edge_list = [None]*int(n_rv*(n_rv-1)/2)
    ind = -1
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

            ind += 1
            edge_list[ind] = (i, j, MI)

        if (i+1)%1000==0:
            print("MI computed for %d variables..." % (i+1))

    return edge_list
    
def mutual_information(data, conditional=False):
    #bins = np.amax(data, axis=0)+1 # read levels for each variable
    # pdb.set_trace()
    bins = unique_bins(data)
        
    if len(bins) == 2:
        hist,_ = np.histogramdd(data, bins=bins[0:2]) # frequency counts

        Pxy = hist / hist.sum()# joint probability distribution over X,Y,Z
        Px = np.sum(Pxy, axis = 1) # P(X,Z)
        Py = np.sum(Pxy, axis = 0) # P(Y,Z) 

        PxPy = np.outer(Px,Py)
        Pxy += 1e-7
        PxPy += 1e-7
        MI = np.sum(Pxy * np.log(Pxy / (PxPy)))
        return round(MI,4)

def unique_bins(data):
    """
    Get the unique values for each column in a dataset.
    """
    bins = np.empty(len(data.T), dtype=np.int32)
    i = 0
    for col in data.T:
        bins[i] = len(np.unique(col))
        i+=1
    return bins

"""
****************
Equivalence Code
****************

This code is for testing whether or not
two Bayesian networks belong to the same
equivalence class - i.e. they have the same
edges when viewed as undirected graphs.

Also, this code is for actually generating
equivalent Bayesian networks from a given BN.

"""

def are_class_equivalent(x,y):
    """
    Check whether two Bayesian networks belong
    to the same equivalence class.
    """
    are_equivalent = True

    if set(list(x.nodes())) != set(list(y.nodes())):
        are_equivalent = False
    else:
        for rv in x.nodes():
            rv_x_neighbors = set(x.parents(rv)) + set(y.children(rv))
            rv_y_neighbors = set(y.parents(rv)) + set(y.children(rv))
            if rv_x_neighbors != rv_y_neighbors:
                are_equivalent =  False
                break
    return are_equivalent