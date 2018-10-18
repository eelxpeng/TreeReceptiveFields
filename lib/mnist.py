import numpy as np
import os
import struct

def load_mnist(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError( "dataset must be 'testing' or 'training'" )

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return img, lbl

if __name__=="__main__":
    trainX, trainY = load_mnist(dataset="training", path="dataset/mnist/")
    print(trainX.shape)
    print(trainX.dtype)
    # print(trainX[0])
    print(trainY[:20])

    testX, testY = load_mnist(dataset="testing", path="dataset/mnist/")
    print(testX.shape)
    print(testX.dtype)
    # print(testX[0])
    print(testY[:20])

