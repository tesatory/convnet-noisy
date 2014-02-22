from convnet import ConvNet
from gpumodel import IGPUModel
from options import *
from data import DataProvider
from noiselayer import NoisyCIFARDataProvider

plot = True
try:
    import pylab as pl
except:
	plot = False

DataProvider.register_data_provider('noisy-cifar', 'Noisy CIFAR', NoisyCIFARDataProvider)
op = ConvNet.get_options_parser()
op.add_option("epochs2", "num_epochs2", IntegerOptionParser, "Number of epochs with noise learning", default=0)
op.add_option("noise-eps", "noise_eps", FloatOptionParser, "Learning rate of noise matrix", default=0.001)
op.add_option("noise-wc", "noise_wc", FloatOptionParser, "Weight cost on noise matrix", default=0.1)

op.options["dp_type"].default = "noisy-cifar"
op.options["data_path"].default = "/home/sainbar/data/cifar-10/train"
op.options["save_path"].default = "/tmp/cifar-10/"
op.options["test_batch_range"].default = [6]
op.options["layer_def"].default = "./example-layers/layers-18pct-noisy.cfg"
op.options["layer_params"].default = "./example-layers/layer-params-18pct-noisy.cfg"
op.options["testing_freq"].default = 10

op, load_dic = IGPUModel.parse_options(op)


model = ConvNet(op, load_dic)
model.start()
if model.num_epochs2 > 0:
	model.libmodel.setNoiseParams(model.noise_eps, model.noise_wc)
	N = model.num_epochs + model.num_epochs2
	while model.num_epochs < N:
		model.num_epochs += 1
		model.start()
		test_errors = [o[0][model.show_cost][model.cost_idx] for o in model.test_outputs]
		pl.plot(test_errors)

model.libmodel.adjustLearningRate(0.1)
model.num_epochs += 10
model.start()
model.libmodel.adjustLearningRate(0.1)
model.num_epochs += 10
model.start()
