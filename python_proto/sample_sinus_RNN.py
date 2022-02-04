import matplotlib.pyplot as plt
import numpy as np
import random
import BeeDNNProto as nn
import Layer as layer


nb_frame = 7
nb_sample = 1000

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
n.append(layer.LayerTimeDistributedDense(1,3))
#n.append(layer.LayerSimplestRNN(3))
n.append(layer.LayerRELU())
n.append(layer.LayerDense(nb_frame*3,nb_frame))

# train net
train = nn.NetTrain()
train.set_epochs(1000)
train.set_batch_size(1000)
train.set_optimizer(nn.opt.OptimizerAdam())
train.set_loss(layer.LossMAE()) # simple Mean Absolute Error
train.fit(n,sample,truth)

# plot loss
plt.plot(train.epoch_loss)
plt.yscale("log")
plt.grid()
plt.title('MAE Loss vs. epochs (logarithmic)')
plt.show()