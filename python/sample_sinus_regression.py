import numpy as np
import matplotlib.pyplot as plt
from beednn import Model, Layer

# Simple Sinus regression using small network
# network is intentionally very small, so we can see the fitting error

# create train data
sample = np.arange(-4.,4.,0.01)[:,np.newaxis]
truth = np.sin(sample)

# construct net
m = Model.Model()
m.append(Layer.LayerDense(1,10))
m.append(Layer.LayerRELU())
m.append(Layer.LayerDense(10,1))

# train net
train = Model.NetTrain()
train.set_epochs(50)
train.set_batch_size(32)
train.set_optimizer("Nadam")
train.set_loss("MSE") # simple Mean Square Error
train.fit(m,sample,truth)

# plot loss
plt.plot(train.epoch_loss)
plt.yscale("log")
plt.grid()
plt.title('MSE Loss vs. epochs (logarithmic)')
plt.show(block=False)

# plot truth curve and predicted
x = sample
y = m.predict(x)
plt.figure()
plt.plot(sample,truth,label='Truth')
plt.plot(sample,y,label='Predicted')
plt.grid()
plt.title('Truth vs. Predicted')
plt.legend()
plt.show()