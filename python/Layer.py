"""
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT license that can be found
    in the LICENSE.txt file.
"""

import numpy as np
import copy
import Optimizer as opt

################################################################################################### Layers
class Layer:
  def __init__(self):
    self.dydx = 1.
    self.training = False # set to False in testing mode or True in training mode
    self.regularizer=None
    self.optimizer=None

  def init(self):
    pass

  def forward(self,x):
    pass

  def set_optimizer(self,optimizer):
    self.optimizer=optimizer

  def optimize(self):
    pass

  #compute input loss from output loss
  def backpropagation(self,dldy):
    return dldy * self.dydx

############################################################################################
class LayerFlatten(Layer):
  def forward(self,x):
    self.input_shape=x.shape[1:]
    return x.reshape((x.shape[0],-1))
     
  def backpropagation(self,dldy):
    return dldy.reshape((-1,self.input_shape[0],self.input_shape[1]))

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

######################################################################################
class LayerGlobalBias(Layer):
  def __init__(self):
    super().__init__()
  
  def init(self):    
    self.bias = np.zeros(1,dtype=np.float32)
    self.optimizer_Bias=opt.create(self.optimizer)

  def forward(self,x):
    return x + self.bias[0]

  def backpropagation(self,dldy):
    self.dldb = np.atleast_1d(dldy.mean())
    return dldy
  
  def optimize(self):
    self.bias=self.optimizer_Bias.optimize(self.bias,self.dldb)
######################################################################################
class LayerBias(Layer):
  def __init__(self,outSize):
    super().__init__()
    self.outSize=outSize
  
  def init(self):
    self.bias = np.zeros((1,self.outSize),dtype=np.float32)
    self.optimizer_Bias=opt.create(self.optimizer)
    
  def forward(self,x):
    return x + self.bias

  def backpropagation(self,dldy):
    self.dldb = np.atleast_2d(np.mean(dldy,axis=0))
    return dldy
  
  def optimize(self):
    self.bias=self.optimizer_Bias.optimize(self.bias,self.dldb)
######################################################################################
class LayerGlobalGain(Layer):
  def __init__(self):
    super().__init__()
  
  def init(self):
    self.weight = 1
    self.optimizer_Weight=opt.create(self.optimizer)
 
  def forward(self,x):
    if self.training:
      self.dydx = self.weight
      self.dydw = x
    return x * self.weight
  
  def backpropagation(self,dldy):
    self.dldw = np.atleast_1d((self.dydw * dldy).mean())
    return dldy * self.dydx

  def optimize(self):
    self.weight=self.optimizer_Weight.optimize(self.weight,self.dldw)
######################################################################################
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
######################################################################################
class LayerDot(Layer):
  def __init__(self,inSize,outSize):
    super().__init__()
    self.inSize=inSize
    self.outSize=outSize

  def init(self):
    a=np.sqrt(6./(self.inSize+self.outSize)) # Xavier uniform initialization
    self.weight = a*(np.random.rand(self.inSize,self.outSize) * 2. - 1.)
    self.optimizer_Weight=opt.create(self.optimizer)

  def forward(self,x):
    if self.training:
      self.dydw = x
    return x @ self.weight

  def backpropagation(self,dldy):
    self.dldw = self.dydw.transpose() @ dldy
    self.dldw *= (1./(self.dydw.shape[0]))
    return dldy @ (self.weight.transpose())

  def optimize(self):
    self.weight=self.optimizer_Weight.optimize(self.weight,self.dldw)
######################################################################################
class LayerDense(Layer): # with bias
  def __init__(self,inSize,outSize):
    super().__init__()
    self.inSize=inSize
    self.outSize=outSize

  def init(self):
    a=np.sqrt(6./(self.inSize+self.outSize)) # Xavier uniform initialization
    self.weight = a*(np.random.rand(self.inSize,self.outSize) * 2. - 1.)
    self.bias = np.zeros((1,self.outSize),dtype=np.float32)
    self.optimizer_Weight=opt.create(self.optimizer)
    self.optimizer_Bias=opt.create(self.optimizer)
 
  def forward(self,x):
    if self.training:
      self.dydw = x
    return (x @ self.weight) + self.bias

  def backpropagation(self,dldy):
    self.dldw = self.dydw.transpose() @ dldy
    self.dldw *= (1./(self.dydw.shape[0]))
    self.dldb = np.atleast_2d(np.mean(dldy,axis=0))
    return dldy @ (self.weight.transpose())

  def optimize(self):
    self.weight=self.optimizer_Weight.optimize(self.weight,self.dldw)
    self.bias=self.optimizer_Bias.optimize(self.bias,self.dldb)
######################################################################################
class LayerSoftmax(Layer):
  def __init__(self):
    super().__init__()

  def forward(self,x):
    max_row=np.max(x,axis=1)
    ex=np.exp(x-max_row[:,None])
    sum_row=np.atleast_2d(np.sum(ex,axis=1)).transpose()
    ex /= sum_row
    
    if self.training:
      self.dydx=ex*(1.-ex)
    return ex
######################################################################################
class LayerDropout(Layer):
  def __init__(self,rate=0.3):
    super().__init__()
    self.rate=rate
 
  def forward(self,x):
    if self.training:
      z = np.random.binomial(size=(1,x.shape[1]), n=1, p= 1-self.rate)*(1./(1.-self.rate))
      x=x*z
    return x
######################################################################################
class LayerTimeDistributedDot(Layer):
  def __init__(self,inSize,outSize):
    super().__init__()
    self.inSize=inSize
    self.outSize=outSize

  def init(self):
    a=np.sqrt(6./(self.inSize+self.outSize)) # Xavier uniform initialization
    self.weight = a*(np.random.rand(self.inSize,self.outSize) * 2. - 1.)
    self.optimizer_Weight=opt.create(self.optimizer)

  def forward(self,x):
    if self.training:
      self.dydw = x

    y=x.reshape((-1,self.inSize))
    d= y @ self.weight
    return d.reshape((x.shape[0],-1))

  def backpropagation(self,dldy):

    # reshape to one frame by row 
    d_reshape=dldy.reshape(-1,self.outSize)

    # compute weight gradient
    self.dydw=self.dydw.reshape(-1,self.inSize)
    self.dldw=self.dydw.T @ d_reshape

    # compute input gradient
    d2= d_reshape @ (self.weight.T)

    # reshape back
    return d2.reshape(dldy.shape[0],-1)

  def optimize(self):
    self.weight=self.optimizer_Weight.optimize(self.weight,self.dldw)
######################################################################################
class LayerTimeDistributedBias(Layer):
  def __init__(self,iSize):
    super().__init__()
    self.iSize=iSize

  def init(self):
    self.bias = np.zeros((1,self.iSize),dtype=np.float32)
    self.optimizer_Bias=opt.create(self.optimizer)
    
  def forward(self,x):
    y=x.reshape((-1,self.iSize))
    d= y + self.bias
    return d.reshape((x.shape[0],-1))

  def backpropagation(self,dldy):

    # reshape to one frame by row 
    d_reshape=dldy.reshape(-1,self.iSize)

     # compute bias gradient
    self.dldb = (np.mean(d_reshape,axis=0,keepdims=True))

    return dldy

  def optimize(self):
    self.bias=self.optimizer_Bias.optimize(self.bias,self.dldb)
######################################################################################
class LayerTimeDistributedDense(Layer):
  def __init__(self,inSize,outSize):
    super().__init__()
    self.inSize=inSize
    self.outSize=outSize

  def init(self):
    a=np.sqrt(6./(self.inSize+self.outSize)) # Xavier uniform initialization
    self.weight = a*(np.random.rand(self.inSize,self.outSize) * 2. - 1.)
    self.bias = np.zeros((1,self.outSize),dtype=np.float32)
    self.optimizer_Weight=opt.create(self.optimizer)
    self.optimizer_Bias=opt.create(self.optimizer)

    
  def forward(self,x):
    if self.training:
      self.dydw = x

    y=x.reshape((-1,self.inSize))
    d= y @ self.weight + self.bias
    return d.reshape((x.shape[0],-1))

  def backpropagation(self,dldy):

    # reshape to one frame by row 
    d_reshape=dldy.reshape(-1,self.outSize)

    # compute weight gradient
    self.dydw=self.dydw.reshape(-1,self.inSize)
    self.dldw=self.dydw.T @ d_reshape

    # compute bias gradient
    self.dldb = (np.mean(d_reshape,axis=0,keepdims=True))

    # compute input gradient
    d2= d_reshape @ (self.weight.T)

    # reshape back
    return d2.reshape(dldy.shape[0],-1)
  
  def optimize(self):
    self.weight=self.optimizer_Weight.optimize(self.weight,self.dldw)
    self.bias=self.optimizer_Bias.optimize(self.bias,self.dldb)
##################################################################################################################################

class RNNCellNaive(): # works like a TimeDistributedDot, no cell memory, toy RNN
  def __init__(self):
    self.h=None

  def init_w(self,inFrameSize,hSize,outFrameSize):
    a=np.sqrt(6./(inFrameSize+outFrameSize)) # Glorot uniform initialization
    self.weight = a*(np.random.rand(inFrameSize,outFrameSize) * 2. - 1.)
    self.hSize=hSize

  def init_state(self):
    self.dldw=np.zeros_like(self.weight)

  def forward(self,x,h):
    if h is None: #init size and zeros
      h=np.zeros((x.shape[0], self.hSize))
    return x@self.weight,0*h #return y,h

  def backpropagation(self,dldy,dldh,x,h):
      self.dldw+= (x.T)@dldy
      dldx=dldy@ (self.weight.T)
      dldh=dldy*0.
      return dldx,dldh #return dldx,dldh

class RNNCellSimplestRNN():
  def __init__(self):
    self.h=None

  def init_w(self,inFrameSize,hSize,outFrameSize):
    a=np.sqrt(6./(inFrameSize+outFrameSize)) # Glorot uniform initialization
    self.weight = a*(np.random.rand(inFrameSize,outFrameSize) * 2. - 1.)
    self.hSize=hSize

  def init_state(self):
    self.dldw=np.zeros_like(self.weight)

  def forward(self,x,h):
    if h is None: #init size and zeros
      h=np.zeros((x.shape[0], self.hSize))
    h=np.tanh( x + h @self.weight) # x can have been affined with a previous TimeDistributedDense
    return h,h #return y,h

  def backpropagation(self,dldy,dldh,x,h):
      dldout=(dldy+dldh)*0.5 #average grad output
      dldu=1.-dldout*dldout #bp tanh

      self.dldw+=(dldu.T)@h

      dldx=dldu
      dldh=dldu@ (self.weight.T)
      return dldx,dldh

##################################################################################################################################

class LayerRNN(Layer): # many to many
  def __init__(self,inFrameSize,hSize,outFrameSize):
    super().__init__()

    self.inFrameSize=inFrameSize
    self.hSize=hSize
    self.outFrameSize=outFrameSize
    #self.cell= RNNCellNaive()
    self.cell= RNNCellSimplestRNN()
    self.cell.init_w(inFrameSize,hSize,outFrameSize)

  def init(self):
    self.optimizer_Weight=opt.create(self.optimizer)
    self.optimizer_Bias=opt.create(self.optimizer)


  def forward(self,x):
    self.cell.init_state()

    if self.training:
      self.x=x

    hf=None
    nb_frame=int(x.shape[1]/self.inFrameSize)
    nb_sample=x.shape[0]
    hSize=self.cell.hSize
    for n in range(nb_frame):
      xf=x[:,n*self.inFrameSize:(n+1)*self.inFrameSize]
      y,h=self.cell.forward(xf,hf)

      if hf is None: # first init
        self.h=np.zeros((nb_sample,hSize*nb_frame))
        self.y=np.zeros((nb_sample,self.outFrameSize*nb_frame))

      self.h[:,n*hSize:(n+1)*hSize]=h
      self.y[:,n*self.outFrameSize:(n+1)*self.outFrameSize]=y
      hf=h

    return self.y

  def backpropagation(self,dldy):
    nb_frames=int(self.x.shape[1]/self.inFrameSize)
    dldx=np.zeros_like(self.x)
    cols=self.x.shape[1]
    self.dldw=np.zeros((cols,cols))
    dldhf=None
    for n in reversed(range(nb_frames)):

      xf=self.x[:,n*self.inFrameSize:(n+1)*self.inFrameSize]
      dldyf=dldy[:,n*self.outFrameSize:(n+1)*self.outFrameSize]
      hf=self.h[:,n*self.outFrameSize:(n+1)*self.outFrameSize]

      # add the loss gradient from next time
      if dldhf is None:
        dldhf=np.zeros_like(hf)

      #backprop cell
      dldxf,dldhf=self.cell.backpropagation(dldyf,dldhf,xf,hf)
      dldx[:,n*self.inFrameSize:(n+1)*self.inFrameSize]=dldxf

    self.dldw=self.cell.dldw/nb_frames
    self.weight=self.cell.weight
    return dldx
    
  def optimize(self):
    self.weight=self.optimizer_Weight.optimize(self.weight,self.dldw)

##################################################################################################################################
