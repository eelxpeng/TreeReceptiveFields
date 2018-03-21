import numpy as np
import scipy.io
from lib.bayesnet.discretize import discretize

data = scipy.io.loadmat("hidden1.mat")
hidden1 = data["hidden1"]

n, d = hidden1.shape
bins = [5]*d
data = discretize(hidden1, bins=bins, verbose=True)
