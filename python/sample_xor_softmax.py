import numpy as np
import matplotlib.pyplot as plt
from beednn import Model, Layer

# simple XOR classification using softmax (see simple_xor.py for a xor classification without softmax)

# create train data
sample=np.zeros((4,2))
sample[0,0]=0 ; sample[0,1]=0
sample[1,0]=0 ; sample[1,1]=1
sample[2,0]=1 ; sample[2,1]=0
sample[3,0]=1 ; sample[3,1]=1

truth=np.zeros((4),dtype=int)
truth[0]=0
truth[1]=1
truth[2]=1
truth[3]=0

# construct net
m = Model.Model()
m.append(Layer.LayerDense(2,3))
m.append(Layer.LayerTanh())
m.append(Layer.LayerDense(3,2))
m.append(Layer.LayerSoftmax())

# optimize net
train = Model.NetTrain()
train.set_epochs(100)
train.set_batch_size(0) # set to 0 for full batch
train.set_optimizer("RPROPm") # work best with full batch size
train.set_loss("SparseCategoricalCrossEntropy") # use this loss for the softmax and categorical truth
train.fit(m,sample,truth)

# plot loss
plt.plot(train.epoch_loss)
plt.grid()
plt.title('Loss vs. epochs')
plt.show()