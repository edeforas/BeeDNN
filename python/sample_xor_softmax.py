import numpy as np
import matplotlib.pyplot as plt
from beednn import Model, Layer

# simple XOR classification using softmax (see sample_xor.py for a xor classification without softmax)

#create XOR data
x_train=np.array([[0,0],[0,1],[1, 0],[1,1]])
y_train=np.array([0,1,1,0])

# construct net
m = Model.Model()
m.append(Layer.LayerDense(2,3))
m.append(Layer.LayerTanh())
m.append(Layer.LayerDense(3,2))
m.append(Layer.LayerSoftmax())

# optimize net
train = Model.Train()
train.set_epochs(100)
train.set_batch_size(0) # set to 0 for full batch
train.set_optimizer("RPROPm") # work best with full batch size
train.set_loss("SparseCategoricalCrossEntropy") # use this loss for the softmax and categorical truth
train.fit(m,x_train,y_train)

# plot loss
plt.plot(train.epoch_loss)
plt.grid()
plt.title('Loss vs. epochs')
plt.show()