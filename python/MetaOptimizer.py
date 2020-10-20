"""
    Copyright (c) 2019, Etienne de Foras and the respective contributors
    All rights reserved.

    Use of this source code is governed by a MIT license that can be found
    in the LICENSE.txt file.
"""

import numpy as np
import copy
import multiprocessing
import threading
import BeeDNN as nn


###################################################################################################
class MetaOptimizer:
  nb_cpu=multiprocessing.cpu_count()
  nb_tries=1
  max_accuracy=-1
  best_net=None

  def set_nb_cpu(self,nb_cpu):
    self.nb_cpu=nb_cpu

  def set_nb_tries(self,nb_tries):
    self.nb_tries=nb_tries

  def run(self,net,netTrain,train_data,train_truth):

    def cb_epochs(netT):
      if netT.get_current_train_accuracy()>self.max_accuracy:
        self.max_accuracy=netT.get_current_train_accuracy()
        self.best_net=copy.deepcopy(netT.n)
        print("better accuracy: "+str(self.max_accuracy))

    print("Using CPU number: "+str(self.nb_cpu)+" and trying: "+str(self.nb_tries)+" times")
    
    for tr in range(self.nb_tries):
      all_threads = list()
      for i in range(self.nb_cpu):
        n2=copy.deepcopy(net)
        train2=copy.deepcopy(netTrain)
        train2.set_epoch_callback(cb_epochs)
        
        t=threading.Thread(target=train2.fit,args=(n2,train_data,train_truth))
        all_threads.append(t)
        t.start()

      for t in all_threads:
          t.join()