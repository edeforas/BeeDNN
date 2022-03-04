import numpy as np
import BeeDNN as nn
import MNIST_import
import MetaOptimizer as meta
import Layer

# MNIST classification using small network and meta optimizer

# load data
[train_data,train_truth,test_data,test_truth]=MNIST_import.load()
train_data/=256.
test_data/=256.

# construct net with a small size
n = nn.Net()
n.append(Layer.LayerFlatten())
n.append(Layer.LayerDense(28*28,64))
n.append(Layer.LayerDropout(0.2))
n.append(Layer.LayerRELU())
n.append(Layer.LayerDense(64,10))
n.append(Layer.LayerSoftmax())

# set optimizer
train = nn.NetTrain()
train.set_epochs(10)
train.set_batch_size(128)
train.set_test_data(test_data , test_truth)
train.set_optimizer("Adam")
train.set_loss("SparseCategoricalCrossEntropy")
train.set_metrics("accuracy")

# run the meta optimizer, using the optimizer as input
mta=meta.MetaOptimizer()
mta.run(n,train, train_data,train_truth)
n=mta.best_net

# compute and print confusion matrix
predicted = n.predict(train_data)
confmat,accuracy=nn.compute_confusion_matrix(train_truth,predicted,10)
print("Train conf mat:\n",confmat)
print("Final Train accuracy:",accuracy)

predicted = n.predict(test_data)
confmat,accuracy=nn.compute_confusion_matrix(test_truth,predicted,10)
print("\nValid conf mat:\n",confmat)
print("Final Valid accuracy:",accuracy)