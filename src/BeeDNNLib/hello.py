import ctypes as ct
import matplotlib.pyplot as plt
import numpy as np

lib = ct.cdll.LoadLibrary("./BeeDNNLib.dll")

#lib.hello()

def activation(activ_name):
    c_string1 = ct.c_char_p(activ_name.encode('utf-8'))
    lib.activation.argtypes = [ct.c_char_p , ct.c_float]
    lib.activation.restype = ct.c_float

    x=np.zeros(1000)
    y=np.zeros(1000)
    for i in range(0,1000):
        x[i]=(i/100-5)
        y[i]=lib.activation(c_string1,i/100-5)

    return x,y

x,y=activation('Relu')
plt.plot(x,y,label='Relu')

x,y=activation('Swish')
plt.plot(x,y,label='Swish')

x,y=activation('HardSwish')
plt.plot(x,y,label='HardSwish')

plt.legend()
plt.grid()
plt.show()