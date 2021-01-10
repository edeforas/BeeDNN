import numpy as np
import BeeDNNProto as nn

#simple xor classification
def test_xor_classification():
	# create train data
	sample=np.zeros((4,2))
	sample[0,0]=0 ; sample[1,0]=0 ; sample[2,0]=1 ; sample[3,0]=1
	sample[0,1]=0 ; sample[1,1]=1 ; sample[2,1]=0 ; sample[3,1]=1
	truth=np.zeros((4,1))
	truth[0,0]=0 ; truth[1,0]=1  ; truth[2,0]=1 ; truth[3,0]=0

	# construct net
	n = nn.Net()
	n.append(nn.LayerDense(2,5))
	n.append(nn.LayerTanh())
	n.append(nn.LayerDense(5,1))

	# optimize net
	train = nn.NetTrain()
	train.epochs = 100
	train.batch_size=0 #if set to 0, use full batch
	train.set_optimizer(nn.opt.OptimizerRPROPm())  # work best with full batch size
	train.set_loss(nn.LossMSE()) # simple Mean Square Error
	train.fit(n,sample,truth)

	# test convergence
	x = sample
	y = n.predict(x)
	dist=np.abs(y-truth)
	assert np.max(dist)<0.01

	print("test_xor_classification succeded.")