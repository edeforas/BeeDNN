import numpy as np
import matplotlib.pyplot as plt
import BeeDNN as nn

# Simple Sinus regression (fit) using small network
# network is very small so we can see the fitting error

# create train data
sample = np.arange(-4.,4.,0.01)[:,np.newaxis]
truth = np.sin(sample)

# construct net
n = nn.Net()
n.set_classification_mode(False) #because we do regression here
n.append(nn.LayerDense(1,10))
n.append(nn.LayerRELU())
n.append(nn.LayerDense(10,1))

# train net
train = nn.NetTrain()
train.epochs = 100
train.batch_size=32
train.set_optimizer(nn.opt.OptimizerNadam())
train.set_loss(nn.LossMSE()) # simple Mean Square Error
train.train(n,sample,truth)

# plot loss
plt.plot(train.epoch_loss)
plt.yscale("log")
plt.grid()
plt.title('MSE Loss vs. epochs (logarithmic)')
plt.show(block=False)

# plot truth curve and predicted
x = sample
y = n.predict(x)
plt.figure()
plt.plot(sample,truth,label='Truth')
plt.plot(sample,y,label='Predicted')
plt.grid()
plt.title('Truth vs. Predicted')
plt.legend()
plt.show()