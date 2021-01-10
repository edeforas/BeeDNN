# to compare the curves of activations: swish and hardswish
# it create two simple network with one activation layer, and call predict

import matplotlib.pyplot as plt
import numpy as np
import BeeDNN as nn

nnSwish=nn.BeeDNN(1)
nnSwish.add_layer('Swish')
nnSwish.set_classification_mode(0)

nnHardSwish=nn.BeeDNN(1)
nnHardSwish.add_layer('HardSwish')
nnHardSwish.set_classification_mode(0)

x=np.arange(-5,5,0.1,dtype=np.float32)
y=np.zeros_like(x)
yh=np.zeros_like(x)
nnSwish.predict(x,y)
nnHardSwish.predict(x,yh)

plt.plot(x,y,label='Swish')
plt.plot(x,yh,label='HardSwish')
plt.title("Swish vs. HardSwish")
plt.legend()
plt.grid()
plt.show()