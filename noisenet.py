from gpumodel import *
from convnet import ConvNet
from options import *
import numpy as n
import scipy.io
import cPickle

def mix_labels(W, labels):
    N = W.shape[0]
    new_labels = n.zeros((1, labels.shape[1]))
    for i in xrange(labels.shape[1]):
        r = n.random.multinomial(1, W[:,int(labels[0, i])])
        new_labels[0, i] = (r * range(N)).sum()
    return new_labels

class NoiseNet(ConvNet):
    def __init__(self, op, load_dic, dp_params={}):
        filename_options = [['train_batch_range','TR'], ['noise_level','NL'], ['noise_true', 'true'], ['noise_wc','WC']]
        dp_params['multiview_test'] = op.get_value('multiview_test')
        dp_params['crop_border'] = op.get_value('crop_border')
        IGPUModel.__init__(self, "ConvNet", op, load_dic, filename_options, dp_params=dp_params)

    def init_model_lib(self):
        if self.layers[-2]['name'] == 'noise':
            if self.noise_true:
                self.layers[-2]['weights'][0] = n.array(self.noise_W.transpose(), dtype=n.single, order='C')
        ConvNet.init_model_lib(self)

    def init_data_providers(self):
        ConvNet.init_data_providers(self)
        if self.noise_level == 0:
            return
        self.noise_W = n.load('data/mixing-offdiag-2.npy')
        self.noise_W = self.noise_level * self.noise_W + (1 - self.noise_level) * n.eye(self.noise_W.shape[0])
        label_path = './data/cifar-mix2-noisy-labels-nl' + str(self.noise_level)
        if os.path.exists(label_path):
            print 'loading noisy labels from ' + label_path
            f = open(label_path,'rb')
            data = cPickle.load(f)
            f.close()
            for d in self.train_data_provider.data_dic:
                d['labels'] = n.require(data[d['batch_label']], dtype=n.single, requirements='C')
        else:
            print 'saving noisy labels to ' + label_path
            data = dict()
            for d in self.train_data_provider.data_dic:
                d['labels'] = mix_labels(self.noise_W, d['labels'])
                d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')
                data[d['batch_label']] = d['labels']
            f = open(label_path,'wb')
            cPickle.dump(data, f, protocol=-1)
            f.close()
    @classmethod
    def get_options_parser(cls):
        op = ConvNet.get_options_parser()
        op.add_option("epochs2", "num_epochs2", IntegerOptionParser, "Number of epochs with noise learning", default=0)
        op.add_option("noise-eps", "noise_eps", FloatOptionParser, "Learning rate of noise matrix", default=0.001)
        op.add_option("noise-wc", "noise_wc", FloatOptionParser, "Weight cost on noise matrix", default=0.1)
        op.add_option("noise-level", "noise_level", FloatOptionParser, "Amount of incorrect training labels", default=0.0)
        op.add_option("noise-true", "noise_true", BooleanOptionParser, "Use true noise matrix", default=0)

        op.options["dp_type"].default = "cifar"
        op.options["data_path"].default = "/home/sainbar/data/cifar-10/train"
        op.options["save_path"].default = "/tmp/cifar-10/"
        op.options["test_batch_range"].default = [6]
        op.options["layer_def"].default = "./example-layers/layers-18pct-noisy.cfg"
        op.options["layer_params"].default = "./example-layers/layer-params-18pct-noisy.cfg"
        op.options["testing_freq"].default = 10

        return op

    def savemat(self, path):
        self.sync_with_host()
        d = dict()
        for l in self.layers:
            if l.has_key('weights'):
                d['weight_' + l['name']] = l['weights'][0]
                d['bias_' + l['name']] = l['biases']
        scipy.io.savemat(path, d)

if __name__ == "__main__":
    op = NoiseNet.get_options_parser()
    op, load_dic = IGPUModel.parse_options(op)
    model = NoiseNet(op, load_dic)

    if model.num_epochs + model.num_epochs2 > 0:
        model.start()
        if model.num_epochs2 > 0:
            model.libmodel.setNoiseParams(model.noise_eps, model.noise_wc)
            model.num_epochs += model.num_epochs2
            model.start()
        model.libmodel.adjustLearningRate(0.1)
        model.num_epochs += 10
        model.start()
        model.libmodel.adjustLearningRate(0.1)
        model.num_epochs += 10
        model.start()
