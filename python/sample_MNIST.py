import numpy as np
import matplotlib.pyplot as plt
import BeeDNN as nn
import MNIST_import;
import ConfusionMatrix;

# Simple MNIST classification using small network

# load data
[sample,truth_categorical,test_sample,test_truth]=MNIST_import.load()
sample/=256.;
sample = sample.reshape(60000, 28*28)
test_sample/=256.;
test_sample = test_sample.reshape(10000, 28*28)
truth=MNIST_import.to_one_hot(truth_categorical);

# construct net
n = nn.Net()
n.append(nn.LayerDense(28*28,128))
n.append(nn.LayerRELU())
n.append(nn.LayerDense(128,10))
n.append(nn.LayerRELU())
n.append(nn.LayerSoftmax())

# train net
train = nn.NetTrain()
train.epochs = 20
train.batch_size=32
train.set_test_data(test_sample , test_truth)
train.set_optimizer(nn.OptimizerMomentum())
train.set_loss(nn.LossCrossEntropy()) # simple Mean Square Error
train.train(n,sample,truth)
n=train.best_net #todo remove do this in the end of train()

# compute and print confusion matrix
predicted = n.forward(sample)
confmat,accuracy=ConfusionMatrix.compute(truth_categorical,predicted,10);
print("Train conf mat:\n",confmat);
print("Final Train accuracy:",accuracy);

predicted = n.forward(test_sample)
confmat,accuracy=ConfusionMatrix.compute(test_truth,predicted,10);
print("\nTest conf mat:\n",confmat);
print("Final Test accuracy:",accuracy);

# plot loss
plt.plot(train.epoch_loss)
plt.yscale("log")
plt.grid()
plt.title('CrossEntropy Loss vs. epochs (logarithmic)')
plt.show()