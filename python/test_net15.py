import math
import numpy as np
import matplotlib.pyplot as plt

import BeeDNN as nn

#create train data
sample = np.arange(-4.,4.,0.1)[:,np.newaxis]
truth = np.exp(sample)

# construct net
n = nn.net()
n.append(nn.layer_dense(1,5))
n.append(nn.layer_relu())
n.append(nn.layer_dense(5,1))

# construct optimizer
train = nn.net_train()
train.set_optimizer(nn.optimizer_sgd())
train.epochs = 100
train.set_loss(nn.loss_mse())
train.train(n,sample,truth)

#plot loss
plt.plot(train.epoch_loss)
plt.grid()
plt.title('MSE vs. epochs')
plt.show()

#plot truth curve and predicted
x = sample
y = n.forward(x)
plt.plot(sample,truth)
plt.plot(sample,y)
plt.grid()
plt.title('truth vs. predicted')
plt.show()