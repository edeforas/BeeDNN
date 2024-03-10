# compare the activations: tanh, smoothtanh and hardtanh
# -> create three simple network with one activation layer in each, and call predict
# no train yet

import matplotlib.pyplot as plt
import numpy as np
import BeeDNNLoader as nn

nnTanh=nn.BeeDNN(1)
nnTanh.add_layer('Tanh')
nnTanh.set_classification_mode(0)

nnSmoothTanh=nn.BeeDNN(1)
nnSmoothTanh.add_layer('SmoothTanh')
nnSmoothTanh.set_classification_mode(0)

nnHardTanh=nn.BeeDNN(1)
nnHardTanh.add_layer('HardTanh')
nnHardTanh.set_classification_mode(0)
#nnHardTanh.save('HardTanh.json') # uncomment to see the network saved in a json file

x=np.arange(-5,5,0.1,dtype=np.float32)
y=np.zeros_like(x)
yh=np.zeros_like(x)
yh1=np.zeros_like(x)

nnTanh.predict(x,y)
nnSmoothTanh.predict(x,yh)
nnHardTanh.predict(x,yh1)

plt.plot(x,y,label='Tanh')
plt.plot(x,yh,label='SmoothTanh')
plt.plot(x,yh1,label='HardTanh')
plt.title("Tanh vs. SmoothTanh vs. HardTanh")
plt.legend()
plt.grid()
plt.show()