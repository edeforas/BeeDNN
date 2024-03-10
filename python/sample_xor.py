import numpy as np
import matplotlib.pyplot as plt
import BeeDNN as nn
import Layer as layer

# simple xor classification and decision surface output

# create train data
sample=np.zeros((4,2))
sample[0,0]=0 ; sample[1,0]=0 ; sample[2,0]=1 ; sample[3,0]=1
sample[0,1]=0 ; sample[1,1]=1 ; sample[2,1]=0 ; sample[3,1]=1
truth=np.zeros((4,1))
truth[0,0]=0 ; truth[1,0]=1  ; truth[2,0]=1 ; truth[3,0]=0

# build net
n = nn.Net()
n.append(layer.LayerDense(2,10))
n.append(layer.LayerRELU())
n.append(layer.LayerDense(10,1))

# train net
train = nn.NetTrain()
train.set_epochs(100)
train.set_batch_size(0) #if set to 0, use full batch
train.set_optimizer("RPROPm")
train.set_loss("MSE") # simple Mean Square Error
train.fit(n,sample,truth)

# plot loss vs. epoch
plt.plot(train.epoch_loss)
plt.grid()
plt.title('Mean Square error vs. epochs')

# compute predicted surface vs X,Y
x=np.linspace(-0.5,1.5,100) #extrapole +/-0.5
y=np.linspace(-0.5,1.5,100) #extrapole +/-0.5
X,Y=np.meshgrid(x,y)
Xr=np.atleast_2d(np.ravel(X)).transpose()
Yr=np.atleast_2d(np.ravel(Y)).transpose()
XY=np.concatenate((Xr,Yr),axis=1)
Z=n.predict(XY)

#plot smooth decision
Zflat=Z.reshape(X.shape)
plt.figure()
plt.pcolormesh(X,Y,Zflat,shading='auto')
plt.title("Smooth decision surface")
plt.colorbar()
plt.grid()

#plot classification frontier
ZClassification=Zflat>0.5
plt.figure()
plt.pcolormesh(X,Y,ZClassification,shading='auto')
plt.title("Classification surface")
plt.colorbar()
plt.grid()

plt.show()