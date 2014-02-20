from convdata import *
import numpy as np

def initW(name, idx, shape, params):
    assert shape[0] == shape[1]
    return np.eye(shape[0], dtype = np.float32)


class NoisyCIFARDataProvider(CIFARDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        CIFARDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        if test == False:
            for d in self.data_dic:
                d['labels'] = self.mix_labels(d['labels'])
                d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')
    
    def mix_labels(self, labels):
        W = np.load('data/mixing-matrix-2.npy')
        N = W.shape[0]
        new_labels = np.zeros((1, labels.shape[1]))
        for i in xrange(labels.shape[1]):
            r = np.random.multinomial(1, W[:,int(labels[0, i])])
            new_labels[0, i] = (r * range(N)).sum()
        return new_labels