import numpy as np
import matplotlib.pyplot as plt
import BeeDNNProto as nn
import MNIST_import

# Simple MNIST classification using small network

# load data
[train_data,train_truth,test_data,test_truth]=MNIST_import.load()
train_data/=256.
train_data = train_data.reshape(60000, 28*28)
test_data/=256.
test_data = test_data.reshape(10000, 28*28)

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
train.log_console=True # show progress
train.set_test_data(test_data , test_truth)
train.set_optimizer(nn.opt.OptimizerAdam())
train.set_loss(nn.LossCategoricalCrossEntropy())
train.fit(n,train_data,train_truth)
n=train.best_net

# compute and print confusion matrix
predicted = n.predict(train_data)
confmat,accuracy=nn.compute_confusion_matrix(train_truth,predicted,10)
print("Train conf mat:\n",confmat)
print("Final Train accuracy:",accuracy)

predicted = n.predict(test_data)
confmat,accuracy=nn.compute_confusion_matrix(test_truth,predicted,10)
print("\nTest conf mat:\n",confmat)
print("Final Test accuracy:",accuracy)

# plot loss
plt.plot(train.epoch_loss)
plt.yscale("log")
plt.grid()
plt.title('CrossEntropy Loss vs. epochs (logarithmic)')
plt.show()