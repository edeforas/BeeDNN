import numpy as np
import BeeDNN as nn
import MNIST_import
import MetaOptimizer as meta

# MNIST classification using small network and meta optimizer

# load data
[train_sample,truth_categorical,test_data,test_truth]=MNIST_import.load()
train_sample/=256.
train_sample = train_sample.reshape(60000, 28*28)
test_data/=256.
test_data = test_data.reshape(10000, 28*28)
train_truth=nn.to_one_hot(truth_categorical)

# construct net with a small size
n = nn.Net()
n.append(nn.LayerDense(28*28,64))
n.append(nn.LayerDropout(0.2))
n.append(nn.LayerRELU())
n.append(nn.LayerDense(64,10))
n.append(nn.LayerSoftmax())

# construct optimizer but do not run
train = nn.NetTrain()
train.epochs = 10
train.log_console=True; # show progress
train.batch_size=64
train.set_test_data(test_data , test_truth)
train.set_optimizer(nn.opt.OptimizerAdam())
train.set_loss(nn.LossCrossEntropy()) # simple Mean Square Error

# construct and run the meta optimizer, using the optimizer as input
mta=meta.MetaOptimizer()
mta.run(n,train, train_sample,train_truth)
n=mta.best_net

# compute and print confusion matrix
predicted = n.predict(train_sample)
confmat,accuracy=nn.compute_confusion_matrix(truth_categorical,predicted,10)
print("Train conf mat:\n",confmat)
print("Final Train accuracy:",accuracy)

predicted = n.predict(test_data)
confmat,accuracy=nn.compute_confusion_matrix(test_truth,predicted,10)
print("\nTest conf mat:\n",confmat)
print("Final Test accuracy:",accuracy)