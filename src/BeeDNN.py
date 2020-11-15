import ctypes as ct
import numpy as np

class BeeDNN:
    c_float_p = ct.POINTER(ct.c_float)
    lib = ct.cdll.LoadLibrary("./BeeDNNLib") # .dll is added under windows, .so under linux
    lib.create.argtypes=[ct.c_int32]
    lib.create.restype=ct.c_void_p
    lib.add_layer.argtypes = [ct.c_void_p,ct.c_char_p]
    lib.set_classification_mode.argtypes = [ct.c_void_p,ct.c_int32]
    lib.predict.argtypes=[ct.c_void_p,c_float_p,c_float_p,ct.c_int32]
        
    def __init__(self,inputSize):
        self.net = ct.c_void_p(self.lib.create(inputSize))
        self.inputSize=inputSize

    def add_layer(self,layer_name):
        cstr = ct.c_char_p(layer_name.encode('utf-8'))
        self.lib.add_layer(self.net,cstr)

    def set_classification_mode(self,bClassificationMode):
        self.lib.set_classification_mode(self.net,ct.c_int32(bClassificationMode))
 
    def predict(self,mIn,mOut):
        data_in = mIn.astype(np.float32)
        nbSamples=ct.c_int32(mIn.shape[0])
       # mOut=np.zeros((mIn.shape[0],1),dtype=np.float32) #todo
        data_p_in = data_in.ctypes.data_as(self.c_float_p)
        data_p_out = mOut.ctypes.data_as(self.c_float_p)
 
        self.lib.predict(self.net,data_p_in,data_p_out,nbSamples)
