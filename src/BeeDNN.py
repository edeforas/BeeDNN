import ctypes as ct
import matplotlib.pyplot as plt
import numpy as np


class BeeDNN:
    c_float_p = ct.POINTER(ct.c_float)
    lib = ct.cdll.LoadLibrary("./BeeDNNLib") #.dll is added under windows, .so under linux
    lib.create.restype=ct.c_void_p
    lib.add_layer.argtypes = [ct.c_void_p,ct.c_char_p]
    lib.set_classification_mode.argtypes = [ct.c_void_p,ct.c_int32]
    lib.predict.argtypes=[ct.c_void_p,c_float_p,c_float_p]
        
    def __init__(self):

        self.net = ct.c_void_p(self.lib.create())

    def add_layer(self,layer_name):
        cstr = ct.c_char_p(layer_name.encode('utf-8'))
        self.lib.add_layer(self.net,cstr)

    def set_classification_mode(self,bClassificationMode):
        self.lib.set_classification_mode(self.net,ct.c_int32(bClassificationMode))
 
    def predict(self,mIn,mOut):
        data_in = mIn.astype(np.float32)
        data_p_in = data_in.ctypes.data_as(self.c_float_p)
        data_p_out = mOut.ctypes.data_as(self.c_float_p)
        self.lib.predict(self.net,data_p_in,data_p_out)
 

nn=BeeDNN()
nn.add_layer('Swish')
nn.set_classification_mode(0)

nnHard=BeeDNN()
nnHard.add_layer('HardSwish')
nnHard.set_classification_mode(0)

mIn=np.zeros((1,1),dtype=np.float32)
mOut=np.zeros((1,1),dtype=np.float32)

xt=np.zeros(0)
yt=np.zeros(0)
yth=np.zeros(0)

for x in np.arange(-5,5,0.1):
    xt=np.append(xt,x)
    mIn[0]=x

    nn.predict(mIn,mOut)
    y=mOut[0]
    yt=np.append(yt,y)

    nnHard.predict(mIn,mOut)
    yh=mOut[0]
    yth=np.append(yth,yh)

plt.plot(xt,yt,label='Swish')
plt.plot(xt,yth,label='HardSwish')
plt.title("Swish vs. HardSwish")
plt.legend()
plt.grid()
plt.show()