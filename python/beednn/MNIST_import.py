#from https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python

import numpy as np

def load():

	f = open('train-images.idx3-ubyte',mode='rb')	
	f.read(16)
	buf = f.read(28 * 28 * 60000)
	train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
	train_data = train_data.reshape(60000, 28, 28)
	f.close()

	f = open('train-labels.idx1-ubyte',mode='rb')
	f.read(8)
	buf = f.read( 60000)
	train_label = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
	f.close()

	f = open('t10k-images.idx3-ubyte',mode='rb')
	f.read(16)
	buf = f.read(28 * 28 * 10000)
	test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
	test_data = test_data.reshape(10000, 28, 28)
	f.close()

	f = open('t10k-labels.idx1-ubyte',mode='rb')
	f.read(8)
	buf = f.read( 10000)
	test_label = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
	f.close()

	return train_data,train_label,test_data,test_label