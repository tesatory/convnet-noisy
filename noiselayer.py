from convdata import *
import numpy as np

def initW(name, idx, shape, params):
    assert shape[0] == shape[1]
    return np.eye(shape[0], dtype = np.float32)