import matplotlib.pyplot as plt
import numpy as np
import random
import BeeDNN as nn
import Layer

nb_frame = 7
nb_sample = 1000
nb_epochs = 5000
batch_size = 1000

#####################################################################################
def create_sample_sin():
    start_sin=random.random()*10-1. ;   end_sin=random.random()*2+1.
    t=np.linspace(start_sin,start_sin+end_sin,nb_frame+1) ;   y=np.sin(t)
    x=y[0:-1] ;   y=y[1:]
    x=np.expand_dims(x,0) ;   y=np.expand_dims(y,0)
    return x, y
#####################################################################################
def create_sample_sin_db():
    x=None ;  y=None
    
    for i in range(nb_sample):
        xs,ys=create_sample_sin()
        
        if x is None:
            x=xs ; y=ys
        else:
            x=np.vstack((x,xs)) ; y=np.vstack((y,ys))
    
    return x,y
#####################################################################################

sample,truth=create_sample_sin_db()
 
n = nn.Net()
n.append(Layer.LayerTimeDistributedDense(1,3))
#n.append(Layer.LayerRNN(3,3,3)) # input frame size, state size, output frame size
n.append(Layer.LayerRELU())
n.append(Layer.LayerDense(3*nb_frame,nb_frame))

# train net
train = nn.NetTrain()
train.set_epochs(nb_epochs)
train.set_batch_size(batch_size)
train.set_optimizer("Adam")
train.set_loss("MAE") # simple Mean Absolute Error
train.fit(n,sample,truth)

# plot loss
plt.plot(train.epoch_loss)
plt.yscale("log")
plt.grid()
plt.title('MAE Loss vs. epochs (logarithmic)')
plt.show()