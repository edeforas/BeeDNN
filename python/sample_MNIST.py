import matplotlib.pyplot as plt
from beednn import MNIST_import, Model, Layer


# Simple MNIST classification using a dense network

# load data
[train_data,train_truth,test_data,test_truth]=MNIST_import.load()
train_data/=256.
test_data/=256.

# construct net
m = Model.Model()
m.append(Layer.LayerFlatten())
m.append(Layer.LayerDense(28*28,128))
m.append(Layer.LayerDropout(0.2))
m.append(Layer.LayerRELU())
m.append(Layer.LayerDense(128,10))
m.append(Layer.LayerSoftmax())

# train net
train = Model.Train()
train.set_epochs(20)
train.set_batch_size(128)
train.set_test_data(test_data , test_truth)
train.set_optimizer("Adam")
train.set_loss("SparseCategoricalCrossEntropy")
train.set_metrics("accuracy")
train.fit(m,train_data,train_truth)
m=train.best_net

# plot loss
plt.plot(train.epoch_train_accuracy,label='train_accuracy')
plt.plot(train.epoch_valid_accuracy,label='valid_accuracy')
plt.legend()
plt.grid()
plt.title('Accuracy vs. epochs')
plt.show()