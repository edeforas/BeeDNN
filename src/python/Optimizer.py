"""
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT license that can be found
    in the LICENSE.txt file.
"""

import numpy as np
import copy

################################################################################################### optimizers
class Optimizer:
  def optimize(self,w,dw):
    pass

#gradient descent algorithm using only gradient sign
class OptimizerStep(Optimizer):
  lr = 0.01
  def optimize(self,w,dw):
    return w - np.sign(dw) * self.lr

#classical gradient descent algorithm
class OptimizerSGD(Optimizer):
  lr = 0.01
  def optimize(self,w,dw):
    return w - dw * self.lr
    
# Momentum from https://cs231n.github.io/neural-networks-3/#sgd
# simplification of below Ng momentum, (avoid one mult)
class OptimizerMomentum(Optimizer):
  lr = 0.01 # is 0.1*(1-momentum)
  momentum = 0.9
  init = False
  def optimize(self,w,dw):
    if not self.init:
      self.v = dw
      self.init = True

    self.v = self.v * self.momentum + dw * self.lr
    return w - self.v

# Andrew Ng definition of momentum
class OptimizerMomentumNg(Optimizer):
  lr = 0.1 # can be high since gradient is smoothed
  momentum = 0.9
  init = False
  def optimize(self,w,dw):
    if not self.init:
      self.v = dw
      self.init = True

    self.v = self.v * self.momentum + dw * (1.-self.momentum) #recursive averaging
    return w - self.v*self.lr # sgd update step with smoothed gradient

# Nesterov from https://cs231n.github.io/neural-networks-3/#sgd
class OptimizerNesterov(Optimizer):
  lr = 0.01
  momentum = 0.9
  init = False
  def optimize(self,w,dw):
    if not self.init:
      self.v = 0. * w
      self.init = True

    v_prev = self.v # back this up
    self.v = self.momentum * self.v - self.lr * dw # velocity update stays the same
    w += -self.momentum * v_prev + (1. + self.momentum) * self.v # position update changes form
    return w

# RPROP-  as in : https://pdfs.semanticscholar.org/df9c/6a3843d54a28138a596acc85a96367a064c2.pdf
# or in paper : Improving the Rprop Learning Algorithm (Christian Igel and Michael Husken)
class OptimizerRPROPm(Optimizer):
  init = False
  
  def optimize(self,w,dw):
    if not self.init:
      self.mu = 0.0125 * np.ones(w.shape)
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

#from https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
class OptimizerAdam(Optimizer): #Adam, with first step bias correction
  beta1=0.9;
  beta2=0.999;
  lr = 0.001
  epsilon=1.e-8
  init = False

  def optimize(self,w,dw):
    if not self.init:
      self.v = 0. * dw
      self.m=self.v
      self.beta1_prod=self.beta1
      self.beta2_prod=self.beta2
      self.init = True

    self.m = self.m*self.beta1+(1.-self.beta1)*dw
    self.v = self.v*self.beta2+(1.-self.beta2)*(dw**2)
    w -= self.lr/(1.-self.beta1_prod) * self.m / (np.sqrt(self.v/(1.-self.beta2_prod)) + self.epsilon);
    self.beta1_prod=self.beta1_prod*self.beta1
    self.beta2_prod=self.beta2_prod*self.beta2
    return w

class OptimizerAdamW(Optimizer):
  beta1=0.9;
  beta2=0.999;
  lr = 0.01
  epsilon=1.e-8
  init = False
  lambda_regul=0.000001

  def optimize(self,w,dw):
    if not self.init:
      self.v = 0. * dw
      self.m=self.v
      self.beta1_prod=self.beta1
      self.beta2_prod=self.beta2
      self.init = True

    self.m = self.m*self.beta1+(1.-self.beta1)*dw
    self.v = self.v*self.beta2+(1.-self.beta2)*(dw**2)
    w -= self.lr/(1.-self.beta1_prod) * self.m / (np.sqrt(self.v/(1.-self.beta2_prod)) + self.epsilon)+self.lambda_regul*w;
    self.beta1_prod=self.beta1_prod*self.beta1
    self.beta2_prod=self.beta2_prod*self.beta2
    return w

class OptimizerAdamax(Optimizer):
  beta1=0.9;
  beta2=0.999;
  lr = 0.01
  epsilon=1.e-8
  init = False

  def optimize(self,w,dw):
    if not self.init:
      self.v = 0. * dw
      self.m=self.v
      self.beta1_prod=self.beta1
      self.init = True

    self.m = self.m*self.beta1+(1.-self.beta1)*dw
    self.v = np.maximum(self.beta2 * self.v, np.abs(dw))+self.epsilon
    w -= self.lr/(1.-self.beta1_prod) * self.m / self.v
    self.beta1_prod=self.beta1_prod*self.beta1
    return w

class OptimizerNadam(Optimizer):
  beta1=0.9;
  beta2=0.999;
  lr = 0.01
  epsilon=1.e-8
  init = False

  def optimize(self,w,dw):
    if not self.init:
      self.v = 0. * dw
      self.m=self.v
      self.beta1_prod=self.beta1
      self.beta2_prod=self.beta2
      self.init = True

    self.m = self.m*self.beta1+(1.-self.beta1)*dw
    self.v = self.v*self.beta2+(1.-self.beta2)*(dw**2)

    m_hat = self.m / (1 - self.beta1_prod) + (1 - self.beta1) * dw / (1 - self.beta1_prod)
    w -=  self.lr * m_hat / (np.sqrt(self.v / (1 - self.beta2_prod)) + self.epsilon)

    self.beta1_prod=self.beta1_prod*self.beta1
    self.beta2_prod=self.beta2_prod*self.beta2
    return w

class OptimizerAmsgrad(Optimizer):
  beta1=0.9;
  beta2=0.999;
  lr = 0.01
  epsilon=1.e-8
  init = False

  def optimize(self,w,dw):
    if not self.init:
      self.v = 0. * dw
      self.v_hat=self.v
      self.m=self.v
      self.init = True

    self.m = self.m*self.beta1+(1.-self.beta1)*dw
    self.v = self.v*self.beta2+(1.-self.beta2)*(dw**2)
    self.v_hat = np.maximum(self.v, self.v_hat)
    w -= self.lr * self.m / (np.sqrt(self.v_hat) + self.epsilon)
    return w

