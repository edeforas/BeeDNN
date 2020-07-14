import numpy as np;

def compute(truth,predicted,nb_class):
	confmat=np.zeros((nb_class,nb_class),dtype=int);
	np.add.at(confmat, tuple([truth.ravel(),predicted.ravel()]), 1)
	accuracy=100.*np.trace(confmat)/np.sum(confmat)
	return confmat,accuracy