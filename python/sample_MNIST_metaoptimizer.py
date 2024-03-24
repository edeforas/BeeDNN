import numpy as np
import beednn as nn

from beednn import Model, Layer, MetaOptimizer,MNIST_import

# MNIST classification using small network and meta optimizer

# load data
[train_data,train_truth,test_data,test_truth]=MNIST_import.load()
train_data/=256.
test_data/=256.

# construct net with a small size
m = Model.Model()
m.append(Layer.LayerFlatten())
m.append(Layer.LayerDense(28*28,64))
m.append(Layer.LayerDropout(0.2))
m.append(Layer.LayerRELU())
m.append(Layer.LayerDense(64,10))
m.append(Layer.LayerSoftmax())

# set optimizer
train = Model.Train()
train.set_epochs(10)
train.set_batch_size(128)
train.set_test_data(test_data , test_truth)
train.set_optimizer("Adam")
train.set_loss("SparseCategoricalCrossEntropy")
train.set_metrics("accuracy")

# run the meta optimizer, using the optimizer as input
mta=MetaOptimizer.MetaOptimizer()
mta.run(m,train, train_data,train_truth)
m=mta.best_net

# compute and print confusion matrix
predicted = m.predict(train_data)
confmat,accuracy=nn.compute_confusion_matrix(train_truth,predicted,10)
print("Train confusion matrix:\n",confmat)
print("Final Train accuracy:",accuracy)

predicted = m.predict(test_data)
confmat,accuracy=Model.compute_confusion_matrix(test_truth,predicted,10)
print("\nValidation confusion matrix:\n",confmat)
print("Final Validation accuracy:",accuracy)