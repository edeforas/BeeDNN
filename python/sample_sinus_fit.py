import math
import numpy as np
import matplotlib.pyplot as plt

import BeeDNN as nn

# Simple Sinus regression ( fit) using small network

# create train data
sample = np.arange(-4.,4.,0.01)[:,np.newaxis]
truth = np.sin(sample)

# construct net
n = nn.Net()
n.classification_mode=False
n.append(nn.LayerDense(1,20))
n.append(nn.LayerRELU())
n.append(nn.LayerDense(20,1))

# optimize net
train = nn.NetTrain()
train.epochs = 100
train.batch_size=32
train.set_optimizer(nn.OptimizerMomentum())
train.set_loss(nn.LossMSE()) # simple Mean Square Error
train.train(n,sample,truth)

# plot loss
plt.plot(train.epoch_loss)
plt.grid()
plt.title('MSE Loss vs. epochs')
plt.show(block=False)

# plot truth curve and predicted
x = sample
y = n.forward(x)
plt.figure()
plt.plot(sample,truth)
plt.plot(sample,y)
plt.grid()
plt.title('Truth vs. Predicted')
plt.show()