from convdata import *
import numpy as np

def initW(name, idx, shape, params):
    assert shape[0] == shape[1]
    # W = np.load('data/mixing-matrix-2.npy')
    # W = W.transpose()
    # return W.astype(np.float32)
    return np.eye(shape[0], dtype = np.float32)
    # return np.random.rand(10,10).astype(np.float32)