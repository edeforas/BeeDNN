import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from beednn import TfUtils

#create datas (load, reshape and normalize)
x_train=np.array([[0,0],[0,1],[1, 0],[1,1]])
y_train=np.array([0,1,1,0])

#create net
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(1))

#train
model.compile(loss='mse', optimizer='adam')
history=model.fit(x_train, y_train, epochs=600)
loss_train=history.history["loss"]
model.summary()

# plot results
plt.plot(loss_train)
plt.title('mse loss')
plt.grid()

y_pred = model.predict(x_train)
plt.figure()
plt.plot(y_train.flatten(),label='y_train')
plt.plot(y_pred.flatten(),label='y_pred')
plt.legend()
plt.grid()
plt.title('truth vs. pred')
plt.show()

#save for inspection
TfUtils.save_tf_to_json(model,'saved_model_xor.json')