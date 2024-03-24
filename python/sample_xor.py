import numpy as np
import matplotlib.pyplot as plt
from beednn import Model, Layer

# simple xor classification and decision surface output

#create XOR data
x_train=np.array([[0,0],[0,1],[1, 0],[1,1]])
y_train=np.array([0,1,1,0])

# build net
m = Model.Model()
m.append(Layer.LayerDense(2,10))
m.append(Layer.LayerRELU())
m.append(Layer.LayerDense(10,1))

# train net
train = Model.Train()
train.set_epochs(200)
train.set_optimizer("RPROPm") # work best with full batch size
train.set_loss("MSE") # Mean Square Error
train.fit(m,x_train,y_train)

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
Z=m.predict(XY)

#plot smooth decision
Zflat=Z.reshape(X.shape)
plt.figure()
plt.pcolormesh(X,Y,Zflat,shading='auto')
plt.title("Smooth decision surface")
plt.colorbar()
plt.grid()

#plot classification surface
ZClassification=Zflat>0.5
plt.figure()
plt.pcolormesh(X,Y,ZClassification,shading='auto')
plt.title("Classification surface")
plt.colorbar()
plt.grid()

plt.show()