import numpy as np
import matplotlib.pyplot as plt
import BeeDNN as nn
import MNIST_import
import Layer

# Simple MNIST classification using a small network

# load data
[train_data,train_truth,test_data,test_truth]=MNIST_import.load()
train_data/=256.
test_data/=256.

# construct net
n = nn.Net()
n.append(Layer.LayerFlatten())
n.append(Layer.LayerDense(28*28,128))
n.append(Layer.LayerDropout(0.2))
n.append(Layer.LayerRELU())
n.append(Layer.LayerDense(128,10))
n.append(Layer.LayerSoftmax())

# train net
train = nn.NetTrain()
train.set_epochs(10)
train.set_batch_size(128)
train.set_test_data(test_data , test_truth)
train.set_optimizer("Adam")
train.set_loss("SparseCategoricalCrossEntropy")
train.set_metrics("accuracy")
train.fit(n,train_data,train_truth)
n=train.best_net

# plot loss
plt.plot(train.epoch_train_accuracy,label='train_accuracy')
plt.plot(train.epoch_valid_accuracy,label='valid_accuracy')
plt.legend()
plt.grid()
plt.title('Accuracy vs. epochs')
plt.show()