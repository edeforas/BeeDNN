import numpy as np
import BeeDNNProto as nn

# Simple Sinus regression (fit) using small network
# network is very small so we can see the fitting error
# this testis to be used with pytest 

def test_fit():
	# create train data
	sample = np.arange(-4.,4.,0.01)[:,np.newaxis]
	truth = np.sin(sample)

	# construct net
	n = nn.Net()
	n.set_classification_mode(False) #because we do regression here
	n.append(nn.LayerDense(1,20))
	n.append(nn.LayerRELU())
	n.append(nn.LayerDense(20,1))

	# train net
	train = nn.NetTrain()
	train.epochs = 100
	train.batch_size=32
	train.set_optimizer(nn.opt.OptimizerNadam())
	train.set_loss(nn.LossMSE()) # simple Mean Square Error
	train.fit(n,sample,truth)

	# test convergence
	x = sample
	y = n.predict(x)
	dist=np.abs(y-truth)
	assert np.max(dist)<0.15