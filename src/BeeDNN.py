import ctypes as ct
import matplotlib.pyplot as plt
import numpy as np


class BeeDNN:
    c_float_p = ct.POINTER(ct.c_float)
    lib = ct.cdll.LoadLibrary("./BeeDNNLib.dll")
    lib.create.restype=ct.c_void_p
    lib.add_layer.argtypes = [ct.c_void_p,ct.c_char_p]
    lib.predict.argtypes=[ct.c_void_p,c_float_p,c_float_p]


    lib.activation.argtypes = [ct.c_char_p , ct.c_float]
    lib.activation.restype = ct.c_float
    net = ct.c_void_p(lib.create())

    def add_layer(self,layer_name):
        cstr = ct.c_char_p(layer_name.encode('utf-8'))
        self.lib.add_layer(self.net,cstr)

    def activation(self,activ_name):
        cstr = ct.c_char_p(activ_name.encode('utf-8'))
        x=np.zeros(1000)
        y=np.zeros(1000)
        for i in range(0,1000):
            x[i]=(i/100-5)
            y[i]=self.lib.activation(cstr,i/100-5)

        return x,y
 
    def predict(self,mIn,mOut):
        data_in = mIn.astype(np.float32)
        data_out = mOut.astype(np.float32)

        data_p_in = data_in.ctypes.data_as(self.c_float_p)
        data_p_out = data_out.ctypes.data_as(self.c_float_p)
        self.lib.predict(self.net,data_p_in,data_p_out)
        pass



nn=BeeDNN()
nn.add_layer('Swish')

mIn=np.zeros((1,1),dtype=float)
mIn[0]=1.234
mOut=np.zeros((1,1),dtype=float)
nn.predict(mIn,mOut)
print(mOut[0])

x,y=nn.activation('Relu')
plt.plot(x,y,label='Relu')

x,y=nn.activation('Swish')
plt.plot(x,y,label='Swish')

x,y=nn.activation('HardSwish')
plt.plot(x,y,label='HardSwish')

plt.legend()
plt.grid()
plt.show()