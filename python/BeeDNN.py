"""
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT license that can be found
    in the LICENSE.txt file.
"""

import math
import numpy as np
import copy

############################## layers
class Layer:
  def __init__(self):
    self.w = 0. 
    self.dydx = 1.
    self.dydw = 0.
    self.learnable = False # set to False to freeze layer or if not learnable
    self.training = False # set to False in testing mode or True in training mode

  def forward(self,x):
    pass

  #compute input loss from output loss
  def backpropagation(self,dldy):
    return dldy * self.dydx

# see https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons
class LayerAbsolute(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = np.sign(x)
    return np.abs(x)
    
# see http://mathworld.wolfram.com/InverseHyperbolicSine.html    
class LayerArcSinh(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = 1. / np.sqrt(1. + x * x)
    return np.arcsinh(x)

class LayerArcTan(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = 1. / (1. + x * x)
    return np.arctan(x)

#see Binary Step as in: https://en.wikipedia.org/wiki/Activation_function
class LayerBinaryStep(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = x * 0.
    return (x > 0.) * 1.

# see https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons
class LayerBipolar(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = x*0.
    return np.sign(x)

#see https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
class LayerBipolarSigmoid(Layer):
  def forward(self,x):
    s=np.exp(x)
    if self.training:
      self.dydx=2.*s/((s+1.)*(s+1.))
    return (s-1.)/(s+1.)

# see https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons
class LayerComplementaryLogLog(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = np.exp(x - np.exp(x))
    return 1. - np.exp(-np.exp(x))

class LayerElliot(Layer):
  def forward(self,x):
    d=1./(1.+np.abs(x))
    if self.training:
        self.dydx = 0.5*d*d
    return 0.5*x*d+0.5

class LayerIdentity(Layer):
  def forward(self,x):
    return x 

class LayerDivideBy256(Layer): # usefull for byte conversion
  def forward(self,x):
    if self.training:
      self.dydx = 0.00390625 # 0.00390625 == 1./256.
    return x*0.00390625

class LayerRELU(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = (x > 0.) * 1.
    return (x > 0.) * x
    
class LayerTanh(Layer):
  def forward(self,x):
    y = np.tanh(x)
    if self.training:
      self.dydx = 1. - y * y
    return y

class LayerTanhShrink(Layer):
  def forward(self,x):
    y = np.tanh(x)
    if self.training:
      self.dydx = y * y
    return x-y

class LayerExponential(Layer):
  def forward(self,x):
    e=np.exp(x)
    if self.training:
      self.dydx = e
    return e

class LayerGauss(Layer):
  def forward(self,x):
    u=np.exp(-x*x)
    if self.training:
      self.dydx = -2.*x*u
    return u

#see Logit as in : https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
class LayerLogit(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = -x/(x-1.)
    return np.log(x/(1.-x))

class LayerLeakyRELU(Layer):
  def forward(self,x):
    if self.training:
      self.dydx=np.ones(x.shape)
      self.dydx[x<0.]=0.01
    u=x #todo cleanup and optimize
    u[u<0.] *= 0.01
    return u

#see LogSigmoid as in : https://nn.readthedocs.io/en/rtd/transfer/
class LayerLogSigmoid(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = 1./(1.+np.exp(x))
    return np.log(1./(1+np.exp(-x)))

class LayerSigmoid(Layer):
  def forward(self,x):
    y = 1. / (1. + np.exp(-x))
    if self.training:
      self.dydx = y * (1. - y)
    return y

class LayerSoftplus(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = 1. / (1. + np.exp(-x))
    return np.log1p(np.exp(x))

# see paper: Swish: A Self-Gated Activation Function
class LayerSwish(Layer):
  def forward(self,x):
    y = 1. / (1. + np.exp(-x))
    if self.training:
      self.dydx = y * (x + 1. - x * y)
    return x * y

# see https://en.wikipedia.org/wiki/Activation_function
class LayerSin(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = np.cos(x)
    return np.sin(x)

####################################### special layers
class LayerGlobalBias(Layer):
  def __init__(self):
    super().__init__()
    self.learnable = True
    self.w = np.zeros(1)
    self.w[0] = 0.

  def forward(self,x):
    return x + self.w[0]
  def backpropagation(self,dldy):
    self.dldw = np.atleast_1d(dldy.mean())
    return dldy

class LayerGlobalGain(Layer):
  def __init__(self):
    super().__init__()
    self.learnable = True
    self.w = np.zeros(1)
    self.w[0] = 1.
 
  def forward(self,x):
    if self.training:
      self.dydx = self.w[0]
      self.dydw = x
    return x * self.w[0]
  
  def backpropagation(self,dldy):
    self.dldw = np.atleast_1d((self.dydw * dldy).mean())
    return dldy * self.dydx

class LayerAddUniformNoise(Layer):
  def __init__(self,noise=0.1):
    super().__init__()
    self.noise=noise
 
  def forward(self,x):
    if self.training:
      x += np.random.rand(*(x.shape))*self.noise
    return x

class LayerDenseNoBias(Layer):
  def __init__(self,inSize,outSize):
    super().__init__()
    self.learnable = True
    a=np.sqrt(6./(inSize+outSize)) # Xavier uniform initialization
    self.w = a*(np.random.rand(inSize,outSize) * 2. - 1.)

  def forward(self,x):
    if self.training:
      self.dydw = x
    return x @ self.w

  def backpropagation(self,dldy):
    self.dldw = self.dydw.transpose() @ dldy
    self.dldw *= (1./(self.dydw.shape[0]))
    return dldy @ (self.w.transpose())

class LayerDense(Layer): # with bias
  def __init__(self,inSize,outSize):
    super().__init__()
    self.learnable = True
    a=np.sqrt(6./(inSize+outSize)) # Xavier uniform initialization
    self.w = a*(np.random.rand(inSize+1,outSize) * 2. - 1.)
 
  def forward(self,x):
    xplusone=np.concatenate((x,np.ones((x.shape[0],1))),axis=1)
    if self.training:
      self.dydw = xplusone
    return xplusone @ self.w

  def backpropagation(self,dldy):
    self.dldw = self.dydw.transpose() @ dldy
    self.dldw *= (1./(self.dydw.shape[0]))
    return dldy @ (self.w[:-1,:].transpose())

class LayerSoftmax(Layer):
  def __init__(self):
    super().__init__()

  def forward(self,x):
    max_row=np.max(x,axis=1)
    ex=np.exp(x-max_row[:,None])
    sum_row=np.atleast_2d(np.sum(ex,axis=1)).transpose()
    if self.training:
      self.dydx=-ex*(ex-sum_row)/(sum_row*sum_row)
    return ex/sum_row

############################ losses
class LayerLoss(Layer):
    def __init__(self):
      super().__init__()
    
    def set_truth(self,truth):
      self.truth = truth

class LossMSE(LayerLoss): #mean square error
  def __init__(self):
    super().__init__()

  def forward(self,x):
    d = x - self.truth
    if self.training:
      self.dydx = d
    return d * d * 0.5

class LossMAE(LayerLoss): #mean absolute error
  def __init__(self):
    super().__init__()
 
  def forward(self,x):
    d = x - self.truth
    if self.training:
      self.dydx = np.sign(d)
    return np.abs(d)

class LossLogCosh(LayerLoss):
  def __init__(self):
    super().__init__()
  
  def forward(self,x):
    d = x - self.truth
    if self.training:
      self.dydx = np.tanh(d)
    return  np.log(np.cosh(d))

# see https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
class LossBinaryCrossEntropy(LayerLoss):
  def __init__(self):
    super().__init__()
  
  def forward(self,x): #todo test x.size ==1
    p=x
    t=self.truth
    if self.training:
      self.dydx = np.atleast_2d(-t/np.maximum(1.e-8,p) +(1.-t)/np.maximum(1.e-8,1.-p) )
    return  np.atleast_2d(-t*np.log(np.maximum(p,1.e-8)) -(1.-t)*np.log(np.maximum(1.e-8,1.-p)))

# see https://gombru.github.io/2018/05/23/cross_entropy_loss
class LossCrossEntropy(LayerLoss):
  def __init__(self):
    super().__init__()
  
  def forward(self,x):
    p_max=np.maximum(x,1.e-8)
    t=self.truth
    if self.training:
      ump_max=np.maximum(1.-x,1.e-8)
      self.dydx = -t/p_max+(1.-t)/ump_max
    return  np.atleast_2d(np.mean(-t*np.log(p_max),axis=1))

############################## optimizers
class Optimizer:
  def optimize(self,w,dw):
    pass

class OptimizerStep(Optimizer):
  lr = 0.01
  def optimize(self,w,dw):
    return w - np.sign(dw) * self.lr

class OptimizerSGD(Optimizer):
  lr = 0.01
  def optimize(self,w,dw):
    return w - dw * self.lr
    
class OptimizerMomentum(Optimizer):
  lr = 0.01
  momentum = 0.9
  init = False
  def optimize(self,w,dw):
    if not self.init:
      self.v = 0. * w
      self.init = True

    self.v = self.v * self.momentum - dw * self.lr
    w = w + self.v
    return w

class OptimizerRPROPm(Optimizer):
  init = False
  
  def optimize(self,w,dw):
    if not self.init:
      self.mu = 0. * w
      self.olddw = dw
      self.init = True
      
    sg = np.sign(dw * self.olddw)
    self.mu[sg < 0.] *= 0.5
    self.mu[sg > 0.] *= 1.2
    np.maximum(self.mu,1.e-6,self.mu)
    np.minimum(self.mu,50.,self.mu)

    w = w - self.mu * np.sign(dw)
    self.olddw = dw
    return w

######################################### net
class Net:
  classification_mode=True
  def __init__(self):
    self.layers = list()

  def append(self,layer):
    self.layers.append(layer)

  def forward(self,x):
    out = x
    for l in self.layers:
      out = l.forward(out)

    if self.classification_mode:
      if out.shape[1]>1:
        out=np.argmax(out,axis=0)
      else:
        out=np.around(out)
    return out

class NetTrain:
  optim = OptimizerMomentum()
  loss_layer = None
  epochs = 100
  batch_size = 32
  n = None

  def set_loss(self,layerloss):
    self.loss_layer = layerloss

  def set_optimizer(self, optim):
    self.optim = optim

  def forward_and_backward(self,x,t):
    #forward pass
    out = x
    for l in self.n.layers:
      out = l.forward(out)

	#backward pass
    self.loss_layer.set_truth(t)
    online_loss = self.loss_layer.forward(out)

    #update and back propagate gradient with loss
    dldy = self.loss_layer.dydx
    for i in range(len(self.n.layers) - 1,-1,-1):
      l = self.n.layers[i]
      dldy = l.backpropagation(dldy)

    return online_loss

  def train(self,n,sample,truth):
    self.n = n
    nblayer = len(n.layers)
    nbsamples = len(sample)
    self.epoch_loss = np.zeros((0,0))

    #init optimizer
    optiml = list()
    for i in range(nblayer):
      optiml.append(copy.copy(self.optim))

    if self.batch_size == 0:
      self.batch_size = nbsamples

    #train!
    for epoch in range(self.epochs):

      #shuffle data
      perm = np.random.permutation(nbsamples)
      s2 = sample[perm,:]
      t2 = truth[perm,:]
      sumloss = 0.

      #set layers in train mode
      self.loss_layer.training=True
      for l in n.layers:
        l.training=True

      batch_start = 0
      while batch_start < nbsamples:
        
        #prepare batch data
        batch_end = min(batch_start + self.batch_size,nbsamples)
        x1 = s2[batch_start:batch_end,:]
        out = t2[batch_start:batch_end,:]
        batch_start += self.batch_size

        #forward pass and gradient computation
        online_loss = self.forward_and_backward(x1,out)
        
        #optimization
        for i in range(len(n.layers)):
          l = n.layers[i]
          if l.learnable:
            l.w = optiml[i].optimize(l.w,l.dldw)

        #save stats
        sumloss+=online_loss.sum()

      #set layers in test mode
      self.loss_layer.training=False
      for l in n.layers:
        l.training=False

      self.epoch_loss = np.append(self.epoch_loss,sumloss / nbsamples)