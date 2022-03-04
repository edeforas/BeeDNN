"""
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT license that can be found
    in the LICENSE.txt file.
"""

import numpy as np
import Layer

###################################################################################################
class LayerLoss(Layer.Layer):
    def __init__(self):
      super().__init__()
    
    def set_truth(self,truth):
      self.truth = truth

    def gradient(self):
      return self.dydx
###################################################################################################
class LossMSE(LayerLoss): #mean square error
  def __init__(self):
    super().__init__()

  def forward(self,x):
    d = x - self.truth
    if self.training:
      self.dydx = d
    return d * d * 0.5
###################################################################################################
class LossMAE(LayerLoss): #mean absolute error
  def __init__(self):
    super().__init__()
 
  def forward(self,x):
    d = x - self.truth
    if self.training:
      self.dydx = np.sign(d)
    return np.abs(d)
###################################################################################################
class LossLogCosh(LayerLoss):
  def __init__(self):
    super().__init__()
  
  def forward(self,x):
    d = x - self.truth
    if self.training:
      self.dydx = np.tanh(d)
    return  np.log(np.cosh(d))
###################################################################################################
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
###################################################################################################
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
###################################################################################################
class LossSparseCategoricalCrossEntropy(LayerLoss):
  def __init__(self):
    super().__init__()
  
  def to_one_hot(self,label,nb_class):
    train_label_one_hot=np.eye(nb_class)[label]
    train_label_one_hot = train_label_one_hot.reshape(label.shape[0], nb_class)
    return train_label_one_hot
    
  def forward(self,x):
    p_max=np.maximum(x,1.e-8)
    
    t=self.to_one_hot(self.truth,x.shape[1]) # todo optimize
    if self.training:
      ump_max=np.maximum(1.-x,1.e-8)
      self.dydx = -t/p_max+(1.-t)/ump_max
    return  np.atleast_2d(np.mean(-t*np.log(p_max),axis=1))

###################################################################################################
def create(sLoss):
  if sLoss=="MSE":
    return LossMSE()

  if sLoss=="MAE":
    return LossMAE()

  if sLoss=="LogCosh":
    return LossLogCosh()

  if sLoss=="SparseCategoricalCrossEntropy":
    return LossSparseCategoricalCrossEntropy()
    
  if sLoss=="CrossEntropy":
    return LossCrossEntropy()

  if sLoss=="BinaryCrossEntropy":
    return LossBinaryCrossEntropy()

  return None
###################################################################################################