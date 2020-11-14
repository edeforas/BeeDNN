"""
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT license that can be found
    in the LICENSE.txt file.
"""

import numpy as np
import copy
import Optimizer as opt

def compute_confusion_matrix(truth,predicted,nb_class=0):
	if nb_class==0:
		nb_class=np.max(truth)
	confmat=np.zeros((nb_class,nb_class),dtype=int)
	np.add.at(confmat, tuple([truth.ravel(),predicted.ravel()]), 1)
	accuracy=100.*np.trace(confmat)/np.sum(confmat)
	return confmat,accuracy

def to_one_hot(label,nb_class):
  train_label_one_hot=np.eye(nb_class)[label]
  train_label_one_hot = train_label_one_hot.reshape(label.shape[0], nb_class)
  return train_label_one_hot

################################################################################################### Layers
class Layer:
  def __init__(self):
    self.dydx = 1.
    self.learnable = False # set to False to freeze layer or if not learnable
    self.training = False # set to False in testing mode or True in training mode
    self.learnBias= False

  def init(self):
    pass

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

class LayerBent(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = x/(2.*np.sqrt(x*x+1.))+1.
    return (np.sqrt(x*x+1.)-1.)*0.5+x

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

class LayerDivideBy256(Layer): # usefull for fixpt code conversion
  def forward(self,x):
    if self.training:
      self.dydx = 0.00390625 # 0.00390625 == 1./256.
    return x*0.00390625

# see Sigmoid-Weighted Linear Units for Neural Network Function ; Stefan Elfwinga Eiji Uchibea Kenji Doyab
class LayerdSiLU(Layer):
  def forward(self,x):
    ex=np.exp(-x)
    exinv=1./(1.+ex)
    if self.training:
      self.dydx = ex*exinv*exinv*(2.+x*(2.*ex*exinv-1.))
    return exinv*(1+x*ex*exinv)

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

# see https://cs224d.stanford.edu/lecture_notes/LectureNotes3.pdf
class LayerHardTanh(Layer):
  def forward(self,x):
    if self.training:
      self.dydx=np.ones(x.shape,dtype=np.float32)
      self.dydx[x<-1.]=0.
      self.dydx[x>1.]=0.
    u=x
    u[x<-1.] = -1.
    u[x>1.] = 1.
    return u

# HardELU, ELU fixed point approximation ; Author is Minh Tri LE
class LayerHardELU(Layer):
  def forward(self,x):
    if self.training:
      self.dydx=np.ones(x.shape,dtype=np.float32)
      self.dydx[ (x<-0.) & (x>-2.) ]=0.5
      self.dydx[ x<-2. ]=0.
    u=x
    u[ (x<-0.) & (x>-2.) ] *= 0.5
    u[x<-2.] = -1.
    return u


# see  https://nn.readthedocs.io/en/rtd/transfer/ (lambda=0.5)	
class LayerHardShrink(Layer):
  def forward(self,x):
    if self.training:
      self.dydx=np.ones(x.shape,dtype=np.float32)
      self.dydx[(x<0.5) & (x>-0.5)]=0.
    u=x
    u[ (x<0.5) & (x>-0.5) ] = 0.
    return u	

class LayerIdentity(Layer):
  def forward(self,x):
    return x 

class LayerLeakyRELU(Layer):
  def forward(self,x):
    if self.training:
      self.dydx=np.ones(x.shape,dtype=np.float32)
      self.dydx[x<0.]=0.01
    u=x
    u[u<0.] *= 0.01
    return u

# easy to convert in fixedpoint (0.00390625= 1/256 = (1>>8) )
class LayerLeakyRELU256(Layer):
  def forward(self,x):
    if self.training:
      self.dydx=np.ones(x.shape,dtype=np.float32)
      self.dydx[x<0.]=0.00390625
    u=x
    u[u<0.] *= 0.00390625
    return u

#see LogSigmoid as in : https://nn.readthedocs.io/en/rtd/transfer/
class LayerLogSigmoid(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = 1./(1.+np.exp(x))
    return np.log(1./(1+np.exp(-x)))

#see Logit as in : https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
class LayerLogit(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = -x/(x-1.)
    return np.log(x/(1.-x))

class LayerRELU(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = (x > 0.) * 1.
    return (x > 0.) * x

class LayerRELU6(Layer):
  def forward(self,x):
    if self.training:
      self.dydx=np.ones(x.shape,dtype=np.float32)
      self.dydx[x<0.]=0.
      self.dydx[x>6.]=0.
    u=x
    u[x<0.] = 0.
    u[x>6.] = 6.
    return u

class LayerSigmoid(Layer):
  def forward(self,x):
    y = 1. / (1. + np.exp(-x))
    if self.training:
      self.dydx = y * (1. - y)
    return y

class LayerSoftPlus(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = 1. / (1. + np.exp(-x))
    return np.log1p(np.exp(x))

class LayerSoftSign(Layer):
  def forward(self,x):
    d=1.+np.abs(x)
    if self.training:
      self.dydx = 1. / (d*d)
    return x/d

# see paper: Swish: A Self-Gated Activation Function
class LayerSwish(Layer):
  def forward(self,x):
    y = 1. / (1. + np.exp(-x))
    if self.training:
      self.dydx = y * (x + 1. - x * y)
    return x * y

# see Sigmoid-Weighted Linear Units for Neural Network Function ; Stefan Elfwinga Eiji Uchibea Kenji Doyab
class LayerSiLU(Layer):
  def forward(self,x):
    ex=np.exp(-x)
    exinv=1./(1.+ex)
    if self.training:
      self.dydx = exinv*(1+x*ex*exinv)
    return x *exinv

# see https://en.wikipedia.org/wiki/Activation_function
class LayerSin(Layer):
  def forward(self,x):
    if self.training:
      self.dydx = np.cos(x)
    return np.sin(x)

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

####################################### special layers
class LayerGlobalBias(Layer):
  def __init__(self):
    super().__init__()
    self.learnBias= True
    self.b = np.zeros(1,dtype=np.float32)

  def forward(self,x):
    return x + self.b[0]

  def backpropagation(self,dldy):
    self.dldb = np.atleast_1d(dldy.mean())
    return dldy

class LayerBias(Layer):
  def __init__(self,outSize):
    super().__init__()
    self.learnBias= True
    self.b = np.zeros((1,outSize),dtype=np.float32)

  def forward(self,x):
    return x + self.b

  def backpropagation(self,dldy):
    self.dldb = np.atleast_2d(np.mean(dldy,axis=0))
    return dldy

class LayerGlobalGain(Layer):
  def __init__(self):
    super().__init__()
    self.learnable = True
    self.w = np.ones(1,dtype=np.float32)
 
  def forward(self,x):
    if self.training:
      self.dydx = self.w[0]
      self.dydw = x
    return x * self.w[0]
  
  def backpropagation(self,dldy):
    self.dldw = np.atleast_1d((self.dydw * dldy).mean())
    return dldy * self.dydx

class LayerAddGaussianNoise(Layer):
  def __init__(self,stdev=0.1):
    super().__init__()
    self.stdev=stdev
 
  def forward(self,x):
    if self.training:
      x += np.random.normal(0, self.stdev,x.shape)
    return x

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
    self.inSize=inSize
    self.outSize=outSize
    self.init()

  def init(self):
    a=np.sqrt(6./(self.inSize+self.outSize)) # Xavier uniform initialization
    self.w = a*(np.random.rand(self.inSize,self.outSize) * 2. - 1.)

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
    self.inSize=inSize
    self.outSize=outSize
    self.learnable = True
    self.learnBias = True
    self.init()

  def init(self):
    a=np.sqrt(6./(self.inSize+self.outSize)) # Xavier uniform initialization
    self.w = a*(np.random.rand(self.inSize,self.outSize) * 2. - 1.)
    self.b = np.zeros((1,self.outSize),dtype=np.float32)
 
  def forward(self,x):
    if self.training:
      self.dydw = x
    return (x @ self.w) + self.b

  def backpropagation(self,dldy):
    self.dldw = self.dydw.transpose() @ dldy
    self.dldw *= (1./(self.dydw.shape[0]))
    self.dldb = np.atleast_2d(np.mean(dldy,axis=0))
    return dldy @ (self.w.transpose())

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

class LayerDropout(Layer):
  def __init__(self,rate=0.3):
    super().__init__()
    self.rate=rate
 
  def forward(self,x):
    if self.training:
      z = np.random.binomial(size=(1,x.shape[1]), n=1, p= 1-self.rate)*(1./(1.-self.rate))
      x=x*z
    return x

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

class LossCategoricalCrossEntropy(LayerLoss):
  def __init__(self):
    super().__init__()
  
  def forward(self,x):
    p_max=np.maximum(x,1.e-8)
    t=to_one_hot(self.truth,x.shape[1]) # todo optimize
    if self.training:
      ump_max=np.maximum(1.-x,1.e-8)
      self.dydx = -t/p_max+(1.-t)/ump_max
    return  np.atleast_2d(np.mean(-t*np.log(p_max),axis=1))

###################################################################################################
class Net:
  classification_mode=True
  def __init__(self):
    self.layers = list()

  def set_classification_mode(self,bClassification):
    self.classification_mode=bClassification

  def append(self,layer):
    self.layers.append(layer)

  def predict(self,x):
    out = x
    for l in self.layers:
      out = l.forward(out)

    if self.classification_mode:
      if out.shape[1]>1:
        out=np.argmax(out,axis=1)
      else:
        out=np.around(out)
      
      out=out.astype(int)
    return out
###################################################################################################
class NetTrain:
  optim = opt.OptimizerMomentum()
  loss_layer = None
  epochs = 100
  batch_size = 32
  test_data=None
  test_truth=None
  log_console=False
  epoch_callback=None
  current_train_accuracy=-1

  #keep best parameters
  keep_best=True
  best_net=None
  best_accuracy=0

  def get_current_train_accuracy(self):
      return self.current_train_accuracy

  def set_loss(self,layerloss):
    self.loss_layer = layerloss

  def set_optimizer(self, optim):
    self.optim = optim

  def set_test_data(self,test_data,test_truth):
    self.test_data=test_data
    self.test_truth=test_truth
 
  def set_keep_best(self,keep_best):
    self.keep_best=keep_best

  def set_epoch_callback(self,epoch_callback):
    self.epoch_callback=epoch_callback

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

  def fit(self,n,train_data,train_truth):
    
    #reshape truth to 2 ndim if 1D
    if train_truth.ndim==1:
      train_truth=np.expand_dims(train_truth, axis=1)

    self.n = n
    self.best_net= copy.deepcopy(n)
    nblayer = len(n.layers)
    nbsamples = len(train_data)
    self.epoch_loss = np.zeros((0,0),dtype=np.float32)

    #init optimizer
    optiml = list()
    for i in range(nblayer*2):
      optiml.append(copy.copy(self.optim)) # reserve weight optimizer
      optiml.append(copy.copy(self.optim)) # reserve bias optimizer

    #init layers
    for l in n.layers:
      l.init()

    if self.batch_size == 0:
      self.batch_size = nbsamples

    #train!
    for epoch in range(self.epochs):
      if self.log_console:
        print("Epoch: ",epoch+1,"/",self.epochs,sep='', end='')

      #shuffle data
      perm = np.random.permutation(nbsamples)
      s2 = train_data[perm,:]
      t2 = train_truth[perm,:]
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
        
        # weights and bias optimization
        for i in range(len(n.layers)):
          l = n.layers[i]
          if l.learnable:
            l.w = optiml[2*i].optimize(l.w,l.dldw)

          if l.learnBias:
            l.b = optiml[2*i+1].optimize(l.b,l.dldb)

        #save stats
        sumloss+=online_loss.sum()

      #set layers in test mode
      self.loss_layer.training=False
      for l in n.layers:
        l.training=False

      self.epoch_loss = np.append(self.epoch_loss,sumloss / nbsamples)

      #compute train accuracy (for now: classification only)
      predicted = n.predict(train_data)

      # classification case
      if(n.classification_mode==True):
        accuracy=100.*np.mean(predicted==train_truth.ravel())
        if self.log_console:
          print(" Train Accuracy: "  +format(accuracy, ".2f")  + "%", end='')

        #compute test accuracy (for now: classification only)
        if self.test_data is not None:
          predicted = n.predict(self.test_data)
          accuracy=100.*np.mean(predicted==self.test_truth.ravel())
          if self.log_console:
            print(" Test Accuracy: "+format(accuracy, ".2f")+"%", end='')
        
        self.current_train_accuracy=accuracy

        if(self.best_accuracy<accuracy):
          self.best_net=copy.deepcopy(n)
          self.best_accuracy=accuracy
          if self.log_console:
            print(" (new best accuracy)",end='')

        if self.log_console:
          print("")
      
      if self.epoch_callback!=None:
        self.epoch_callback(self)

    # if self.keep_best and (self.best_accuracy!=0):
    #   n=copy.deepcopy(self.best_net)