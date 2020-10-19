import numpy as np
import matplotlib.pyplot as plt
import BeeDNN as nn
import MNIST_import

# Simple MNIST classification using small network

# load data
[train_sample,truth_categorical,test_sample,test_truth]=MNIST_import.load()
train_sample/=256.
train_sample = train_sample.reshape(60000, 28*28)
test_sample/=256.
test_sample = test_sample.reshape(10000, 28*28)
truth=nn.to_one_hot(truth_categorical)

# construct net
n = nn.Net()
n.append(nn.LayerDense(28*28,128))
n.append(nn.LayerDropout(0.2))
n.append(nn.LayerRELU())
n.append(nn.LayerDense(128,10))
n.append(nn.LayerSoftmax())

# train net
train = nn.NetTrain()
train.epochs = 20
train.batch_size=64
train.set_test_data(test_sample , test_truth)
train.set_optimizer(nn.opt.OptimizerAdam())
train.set_loss(nn.LossCrossEntropy()) # simple Mean Square Error
train.fit(n,train_sample,truth)
n=train.best_net

# compute and print confusion matrix
predicted = n.predict(train_sample)
confmat,accuracy=nn.compute_confusion_matrix(truth_categorical,predicted,10)
print("Train conf mat:\n",confmat)
print("Final Train accuracy:",accuracy)

predicted = n.predict(test_sample)
confmat,accuracy=nn.compute_confusion_matrix(test_truth,predicted,10)
print("\nTest conf mat:\n",confmat)
print("Final Test accuracy:",accuracy)

# plot loss
plt.plot(train.epoch_loss)
plt.yscale("log")
plt.grid()
plt.title('CrossEntropy Loss vs. epochs (logarithmic)')
plt.show()