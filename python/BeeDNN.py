import math
import numpy as np
import copy


############################## layers
class layer:
  def __init__(self):
    self.w = 0. 
    self.dydx = 1.
    self.dydw = 0.
    self.learnable = False

  def forward(self,x):
    pass

  def forward_and_gradient(self,x):
    pass

  #compute input loss for output loss
  def backpropagation(self,dldy):
    return dldy * self.dydx

class layer_identity(layer):
  def forward(self,x):
      return x 
  def forward_and_gradient(self,x):
    return x

class layer_relu(layer):
  def forward(self,x):
      return (x > 0.) * x
  def forward_and_gradient(self,x):
    self.dydx = (x > 0.) * 1.
    return (x > 0.) * x
    
class layer_tanh(layer):
  def forward(self,x):
      return np.tanh(x)
  def forward_and_gradient(self,x):
    y = np.tanh(x)
    self.dydx = 1. - y * y
    return y

class layer_atan(layer):
  def forward(self,x):
      return np.arctan(x)
  def forward_and_gradient(self,x):
    self.dydx = 1. / (1. + x * x)
    return np.arctan(x)

class layer_complementaryloglog(layer):
  def forward(self,x):
      return 1. - np.exp(-np.exp(x))
  def forward_and_gradient(self,x):
    self.dydx = np.exp(x - np.exp(x))
    return 1. - np.exp(-np.exp(x))

class layer_sigmoid(layer):
  def forward(self,x):
    y = 1. / (1. + np.exp(-x))
    return y
  def forward_and_gradient(self,x):
    y = 1. / (1. + np.exp(-x))
    self.dydx = y * (1. - y)
    return y

class layer_softplus(layer):
  def forward(self,x):
    return np.log1p(np.exp(x))
  def forward_and_gradient(self,x):
    self.dydx = 1. / (1. + np.exp(-x))
    return np.log1p(np.exp(x))

class layer_swish(layer):
  def forward(self,x):
    y = x / (1. + np.exp(-x))
    return y
  def forward_and_gradient(self,x):
    y = 1. / (1. + np.exp(-x))
    self.dydx = y * (x + 1. - x * y)
    return x * y

####################################### special layers
class layer_bias(layer):
  def __init__(self):
    super().__init__()
    self.learnable = True
    self.w = np.zeros(1)
    self.w[0] = 0.

  def forward(self,x):
    return x + self.w[0]
  def forward_and_gradient(self,x):
    return x + self.w[0]
  def backpropagation(self,dldy):
    self.dldw = dldy.mean(axis=0)
    return dldy

class layer_gain(layer):
  def __init__(self):
    super().__init__()
    self.learnable = True
    self.w = np.zeros(1)
    self.w[0] = 1.

  def forward(self,x):
      return x * self.w[0]
  
  def forward_and_gradient(self,x):
    self.dydx = self.w[0]
    self.dydw = x
    return x * self.w[0]
  def backpropagation(self,dldy):
    self.dldw = (self.dydw * dldy).mean(axis=0)
    return dldy * self.dydx

"""
	
	
class layer_affine(layer):
  def __init__(self):
    super().__init__()
    self.learnable=True
    self.w=np.zeros(2)
    self.w[0]=1.
    self.w[1]=0.

  def forward(self,x):
      return x*self.w[0]+self.w[1]
  
  def forward_and_gradient(self,x):
    self.dydx=self.w[0]
    self.dydw=np.concatenate((x,x*0.+1.),axis=1)
    return x*self.w[0]+self.w[1]
	
  def backpropagation(self,dldy):
    self.dldw=self.dydw*dldy
    return dldy * self.dydx
"""
	
	
	
class layer_dense_nobias(layer):
  def __init__(self,inSize,outSize):
    super().__init__()
    self.learnable = True
    self.inSize = inSize
    self.outSize = outSize
    self.w = np.random.rand(inSize,outSize) * 2. - 1.

  def forward(self,x):
    return x @ self.w
  
  def forward_and_gradient(self,x):
    self.dydw = x
    return x @ self.w
  def backpropagation(self,dldy):
    dldw = self.dydw.transpose() @ dldy
    self.dldw = dldw.mean(axis=0)
    return dldy @ (self.w.transpose())


############################ losses
class layer_loss(layer):
    def __init__(self):
      super().__init__()
    
    def set_truth(self,truth):
      pass

class loss_mse(layer_loss):
  def __init__(self):
    super().__init__()

  def set_truth(self,truth):
    self.truth = truth

  def forward(self,x):
    d = x - self.truth
    return d * d * 0.5
  
  def forward_and_gradient(self,x):
    d = x - self.truth
    self.dydx = d
    return d * d * 0.5

class loss_logcosh(layer_loss):
  def __init__(self):
    super().__init__()

  def set_truth(self,truth):
    self.truth = truth

  def forward(self,x):
    d = x - self.truth
    return np.log(np.cosh(d))
  
  def forward_and_gradient(self,x):
    d = x - self.truth
    self.dydx = np.tanh(d)
    return  np.log(np.cosh(d))

############################## optimizers
class optimizer:
  pass

class optimizer_step(optimizer):
  lr = 0.01
  def optimize(self,w,dw):
    return w - np.sign(dw) * self.lr

class optimizer_sgd(optimizer):
  lr = 0.01
  def optimize(self,w,dw):
    return w - dw * self.lr
    
class optimizer_momentum(optimizer):
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

class optimizer_RPROPm(optimizer):
  init = False
  
  def optimize(self,w,dw):
    if not self.init:
      self.mu = 0. * w
      self.olddw = dw
      self.init = True
      
    min_mu = 1.e-6
    max_mu = 50.

    sg = np.sign(dw * self.olddw)
    self.mu[sg < 0.] *= 0.5
    self.mu[sg > 0.] *= 1.2
    self.mu[self.mu < min_mu] = min_mu
    self.mu[self.mu > max_mu] = max_mu

    w = w - self.mu * np.sign(dw)
    self.olddw = dw
    return w

######################################### net
class net:
  def __init__(self):
    self.layers = list()

  def append(self,layer):
    self.layers.append(layer)

  def forward(self,x):
    out = x
    for l in self.layers:
      out = l.forward(out)
    return out

class net_train:
  optim = optimizer_RPROPm()
  loss_layer = None
  epochs = 100
  batch_size = 0
  n = None

  def set_loss(self,layerloss):
    self.loss_layer = layerloss

  def set_optimizer(self, optim):
    self.optim = optim

  def forward_and_backward(self,x,t):
    #forward pass
    out = x
    for l in self.n.layers:
      out = l.forward_and_gradient(out)

	#backward pass
    self.loss_layer.set_truth(t)
    outloss = self.loss_layer.forward_and_gradient(out)

    #update and back propagate gradient with loss
    dldy = self.loss_layer.dydx
    for i in range(len(self.n.layers) - 1,-1,-1):
      l = self.n.layers[i]
      dldy = l.backpropagation(dldy)

    return out,outloss

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

      batch_start = 0
      while batch_start < nbsamples:
        
        #prepare batch data
        batch_end = min(batch_start + self.batch_size,nbsamples)
        x1 = s2[batch_start:batch_end,:]
        out = t2[batch_start:batch_end,:]
        batch_start += self.batch_size

        #forward pass and gradient computation
        predicted,predictedloss = self.forward_and_backward(x1,out)
        
        #optimization
        for i in range(len(n.layers)):
          l = n.layers[i]
          if l.learnable:
            l.w = optiml[i].optimize(l.w,l.dldw)

        #save stats
        sumloss+=predictedloss.sum(axis=0)

      self.epoch_loss = np.append(self.epoch_loss,sumloss / nbsamples)