import numpy as np
import matplotlib.pyplot as plt
import BeeDNN as nn
import Layer

# Simple Sinus regression using small network
# network is intentionally very small, so we can see the fitting error

# create train data
sample = np.arange(-4.,4.,0.01)[:,np.newaxis]
truth = np.sin(sample)

# construct net
n = nn.Net()
n.append(Layer.LayerDense(1,10))
n.append(Layer.LayerRELU())
n.append(Layer.LayerDense(10,1))

# train net
train = nn.NetTrain()
train.set_epochs(50)
train.set_batch_size(32)
train.set_optimizer("Nadam")
train.set_loss("MSE") # simple Mean Square Error
train.fit(n,sample,truth)

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