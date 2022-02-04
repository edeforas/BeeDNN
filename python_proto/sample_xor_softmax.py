import numpy as np
import matplotlib.pyplot as plt
import BeeDNNProto as nn
import Layer as layer

# simple XOR classification using softmax (see simple_xor.py for a xor classification without softmax)

# create train data
sample=np.zeros((4,2))
sample[0,0]=0 ; sample[0,1]=0
sample[1,0]=0 ; sample[1,1]=1
sample[2,0]=1 ; sample[2,1]=0
sample[3,0]=1 ; sample[3,1]=1

truth=np.zeros((4),dtype=int)
truth[0]=0
truth[1]=1
truth[2]=1
truth[3]=0

# construct net
n = nn.Net()
n.append(layer.LayerDense(2,3))
n.append(layer.LayerTanh())
n.append(layer.LayerDense(3,2))
n.append(layer.LayerSoftmax())

# optimize net
train = nn.NetTrain()
train.set_epochs(100)
train.set_batch_size(0) # set to 0 for full batch
train.set_optimizer(nn.opt.OptimizerRPROPm()) # work best with full batch size
train.set_loss(layer.LossSparseCategoricalCrossEntropy()) # use this loss for the softmax and categorical truth
train.fit(n,sample,truth)

# plot loss
plt.plot(train.epoch_loss)
plt.grid()
plt.title('Loss vs. epochs')
plt.show()