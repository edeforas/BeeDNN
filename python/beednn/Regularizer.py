"""
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT license that can be found
    in the LICENSE.txt file.
"""

import numpy as np

################################################################################################### Layers
class Regularizer:
  def __init__(self,alpha):
    self.alpha = alpha

  def apply(self,w,dw):
    pass
   
############################################################################################
# as in : https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/
class RegularizerL2(Regularizer):
  def apply(self,w,dw):
    return w,dw+w*self.alpha
############################################################################################