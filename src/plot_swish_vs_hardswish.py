# to compare the curves of activations: swish and hardswish
# we create two simple network with one activation layer, and call predict

import matplotlib.pyplot as plt
import numpy as np
import BeeDNN as nn

nnSwish=nn.BeeDNN()
nnSwish.add_layer('Swish')
nnSwish.set_classification_mode(0)

nnHardSwish=nn.BeeDNN()
nnHardSwish.add_layer('HardSwish')
nnHardSwish.set_classification_mode(0)

mIn=np.zeros((1,1),dtype=np.float32)
mOut=np.zeros((1,1),dtype=np.float32)

xt=np.zeros(0)
yt=np.zeros(0)
yth=np.zeros(0)

for x in np.arange(-5,5,0.1):
    xt=np.append(xt,x)
    mIn[0]=x

    nnSwish.predict(mIn,mOut)
    y=mOut[0]
    yt=np.append(yt,y)

    nnHardSwish.predict(mIn,mOut)
    yh=mOut[0]
    yth=np.append(yth,yh)

plt.plot(xt,yt,label='Swish')
plt.plot(xt,yth,label='HardSwish')
plt.title("Swish vs. HardSwish")
plt.legend()
plt.grid()
plt.show()