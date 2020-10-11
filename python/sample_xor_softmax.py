import numpy as np
import matplotlib.pyplot as plt
import BeeDNN as nn

# Simple XOR classification using softmax (see simple_xor.py for a xor classification without softmax)

# create train data
sample=np.zeros((4,2))
sample[0,0]=0 ; sample[1,0]=0 ; sample[2,0]=1 ; sample[3,0]=1
sample[0,1]=0 ; sample[1,1]=1 ; sample[2,1]=0 ; sample[3,1]=1
truth=np.zeros((4,2))
truth[0,0]=0 ; truth[1,0]=1  ; truth[2,0]=1 ; truth[3,0]=0
truth[0,1]=1 ; truth[1,1]=0  ; truth[2,1]=0 ; truth[3,1]=1

# construct net
n = nn.Net()
n.append(nn.LayerDense(2,3))
n.append(nn.LayerTanh())
n.append(nn.LayerDense(3,2))
n.append(nn.LayerSoftmax())

# optimize net
train = nn.NetTrain()
train.epochs = 100
train.batch_size=0 # set to 0 for full batch
train.set_optimizer(nn.opt.OptimizerRPROPm()) # work best with full batch size
train.set_loss(nn.LossCrossEntropy()) # use this loss for the softmax
train.fit(n,sample,truth)

# plot loss
plt.plot(train.epoch_loss)
plt.grid()
plt.title('Loss vs. epochs')
plt.show()