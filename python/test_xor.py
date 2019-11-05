import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D  

import BeeDNN as nn

# create train data
sample=np.zeros((4,2))
sample[0,0]=0 ; sample[1,0]=0 ; sample[2,0]=1 ; sample[3,0]=1
sample[0,1]=0 ; sample[1,1]=1 ; sample[2,1]=0 ; sample[3,1]=1

truth=np.zeros((4,1))
truth[0,0]=0 ; truth[1,0]=1  ; truth[2,0]=1 ; truth[3,0]=0

# construct net
n = nn.Net()
n.append(nn.LayerDense(2,5))
n.append(nn.LayerGauss())
n.append(nn.LayerDense(5,1))

# optimize net
train = nn.NetTrain()
train.epochs = 200
train.batch_size=0
train.set_optimizer(nn.OptimizerRPROPm())
train.set_loss(nn.LossMSE())
train.train(n,sample,truth)

# plot loss
plt.plot(train.epoch_loss)
plt.grid()
plt.title('MSE vs. epochs')
plt.show()

# plot predicted in 3D surface, without the last classification step
x=np.linspace(-2,2,20)
y=np.linspace(-2,2,20)
X,Y=np.meshgrid(x,y)
Xr=np.atleast_2d(np.ravel(X)).transpose()
Yr=np.atleast_2d(np.ravel(Y)).transpose()
XY=np.concatenate((Xr,Yr),axis=1)
Z=n.forward(XY)
Z=Z.reshape(X.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z,cmap=cm.coolwarm, antialiased=False)
plt.show()