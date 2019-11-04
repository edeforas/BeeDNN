import math
import numpy as np
import matplotlib.pyplot as plt

import BeeDNN as nn

# create train data
sample = np.arange(-4.,4.,0.01)[:,np.newaxis]
truth = np.sin(sample)

# construct net
n = nn.Net()
n.append(nn.LayerDense(1,20))
n.append(nn.LayerRELU())
n.append(nn.LayerDense(20,1))

# optimize net
train = nn.NetTrain()
train.epochs = 1000
train.batch_size=32
train.set_optimizer(nn.OptimizerMomentum())
train.set_loss(nn.LossMSE())
train.train(n,sample,truth)

# plot loss
plt.plot(train.epoch_loss)
plt.grid()
plt.title('MSE vs. epochs')
plt.show()

# plot truth curve and predicted
x = sample
y = n.forward(x)
plt.plot(sample,truth)
plt.plot(sample,y)
plt.grid()
plt.title('truth vs. predicted')
plt.show()