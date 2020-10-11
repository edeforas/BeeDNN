from sklearn.neural_network  import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import BeeDNN as nn

sample=np.asarray([[0, 0], [0, 1] ,[1, 0] ,[1, 1]])
truth=np.asarray([[0],[1], [1], [0]])

reg = DecisionTreeClassifier()
reg.fit(sample, truth.ravel())
print("Tree:"+str(reg.predict(sample)))

reg = MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000)
reg.fit(sample, truth.ravel())
print("MLP:"+str(reg.predict(sample)))

# BeeDNN build , train and test
n = nn.Net()
n.append(nn.LayerDense(2,3))
n.append(nn.LayerTanh())
n.append(nn.LayerDense(3,1))
train = nn.NetTrain()
train.epochs = 50
train.set_optimizer(nn.opt.OptimizerRPROPm())  # work best with full batch size
train.set_loss(nn.LossMSE()) # simple Mean Square Error
train.fit(n,sample,truth)

print("BeeDNN:"+str(n.predict(sample).ravel()))