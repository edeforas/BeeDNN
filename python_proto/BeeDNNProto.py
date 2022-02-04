"""
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT license that can be found
    in the LICENSE.txt file.
"""

import numpy as np
import copy
import Optimizer as opt
import Layer as layer

def compute_confusion_matrix(truth,predicted,nb_class=0):
  if nb_class==0:
    nb_class=np.max(truth)+1

  if predicted.shape[1]>1:
    predicted=np.argmax(predicted, axis=1)  

  confmat=np.zeros((nb_class,nb_class),dtype=int)
  np.add.at(confmat, tuple([truth.ravel(),predicted.ravel()]), 1)
  accuracy=100.*np.trace(confmat)/np.sum(confmat)
  return confmat,accuracy

def to_one_hot(label,nb_class):
  train_label_one_hot=np.eye(nb_class)[label]
  train_label_one_hot = train_label_one_hot.reshape(label.shape[0], nb_class)
  return train_label_one_hot

###################################################################################################
class Net:
  def __init__(self):
    self.layers = []

  def append(self,layer):
    self.layers.append(layer)

  def init(self):
    for l in self.layers:
      l.init()

  def predict(self,x):
    out = x
    for l in self.layers:
      out = l.forward(out)

    return out
###################################################################################################
class NetTrain:
  optim = opt.OptimizerAdam()
  loss_layer = None
  epochs = 100
  batch_size = 32
  test_data=None
  test_truth=None
  epoch_callback=None
  current_train_accuracy=-1
  metrics=""

  #keep best parameters
  keep_best=True
  best_net=None
  best_accuracy=0

  def get_current_train_accuracy(self):
      return self.current_train_accuracy

  def set_loss(self,layerloss):
    self.loss_layer = layerloss

  def set_epochs(self,epochs):
    self.epochs=epochs

  def set_batch_size(self,batch_size):
    self.batch_size=batch_size

  def set_metrics(self,metrics):
    self.metrics=metrics

  def set_optimizer(self, optim):
    self.optim = optim

  def set_test_data(self,test_data,test_truth):
    self.test_data=test_data
    self.test_truth=test_truth
 
  def set_keep_best(self,keep_best):
    self.keep_best=keep_best

  def set_epoch_callback(self,epoch_callback):
    self.epoch_callback=epoch_callback

  def forward_then_backward(self,x,t):
    #forward pass
    out= self.n.predict(x)

    # compute loss
    self.loss_layer.set_truth(t)
    online_loss = self.loss_layer.forward(out)

	  #backward pass
    grad = self.loss_layer.gradient()
    for l in reversed(self.n.layers):
      grad = l.backpropagation(grad)

    return online_loss

  def fit(self,n,train_data,train_truth):
    
    #reshape truth to 2 ndim if 1D
    if train_truth.ndim==1:
      train_truth=np.expand_dims(train_truth, axis=1)

    self.n = n
    self.best_net= copy.deepcopy(n)
    nblayer = len(n.layers)
    nbsamples = len(train_data)
    self.epoch_loss = np.zeros((0,0),dtype=np.float32)
    self.epoch_train_accuracy = np.zeros((0,0),dtype=np.float32)
    self.epoch_valid_accuracy = np.zeros((0,0),dtype=np.float32)

    #init states
    n.init()
    optiml = []
    for i in range(nblayer*2):
      optiml.append(copy.copy(self.optim)) # reserve weight optimizer
      optiml.append(copy.copy(self.optim)) # reserve bias optimizer

    if self.batch_size == 0:
      self.batch_size = nbsamples

    #set layers in train mode
    self.loss_layer.training=True
    for l in n.layers:
      l.training=True

    #train!
    for epoch in range(self.epochs):
      print("Epoch: "+str(epoch+1)+"/"+str(self.epochs),end='',flush=True)

      #shuffle data
      perm = np.random.permutation(nbsamples)
      s2 = train_data[perm,:]
      t2 = train_truth[perm,:]
      sumloss = 0.

      batch_start = 0
      while batch_start < nbsamples:
        
        #prepare batch data
        batch_end = min(batch_start + self.batch_size,nbsamples)
        x_train = s2[batch_start:batch_end,:]
        y_train = t2[batch_start:batch_end,:]
        batch_start += self.batch_size

        #forward pass and gradient computation
        online_loss = self.forward_then_backward(x_train,y_train)
        
        # weights and bias optimization
        for i in range(len(n.layers)):
          l = n.layers[i]
          if l.learnWeight:
            assert (l.weight.shape==l.dldw.shape),"Weight and Gradient weight should have the same shape"
            l.weight = optiml[2*i].optimize(l.weight,l.dldw)

          if l.learnBias:
            assert (l.bias.shape==l.dldb.shape),"Bias and Gradient bias should have the same shape"
            l.bias = optiml[2*i+1].optimize(l.bias,l.dldb)

        #save stats
        sumloss+=online_loss.sum()

      epoch_loss=sumloss / train_data.size
      self.epoch_loss = np.append(self.epoch_loss,epoch_loss)

      if self.metrics =="accuracy":
        predicted = n.predict(train_data)
        cm,accuracy=compute_confusion_matrix(train_truth,predicted)
        self.current_train_accuracy=accuracy
        self.epoch_train_accuracy = np.append(self.epoch_train_accuracy,accuracy)
        print(" Train Accuracy: "  +format(accuracy, ".2f")  + "%", end='')
        if self.test_data is not None:
          predicted = n.predict(self.test_data)
          cm,accuracy=compute_confusion_matrix(self.test_truth,predicted)
          self.current_valid_accuracy=accuracy
          self.epoch_valid_accuracy = np.append(self.epoch_valid_accuracy,accuracy)
          print(" Valid Accuracy: "+format(accuracy, ".2f")+"%", end='')

          if(self.best_accuracy<accuracy):
            self.best_net=copy.deepcopy(n)
            self.best_accuracy=accuracy
            print(" (new best accuracy)",end='')
        
        print("")
      else:
        print(" Loss: "+str(epoch_loss)+ " ",flush=True)

      if self.epoch_callback!=None:
        self.epoch_callback(self)

    #set layers in test mode
    self.loss_layer.training=False
    for l in n.layers:
      l.training=False

    # if self.keep_best and (self.best_accuracy!=0):
    #   n=copy.deepcopy(self.best_net)