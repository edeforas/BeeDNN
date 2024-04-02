import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from beednn import TFUtils,BeeDNNUtils

#create datas (load, reshape and normalize)
sunspot="sunspot.txt"
spot=np.loadtxt(sunspot)
max_spot=spot.max()
spot=spot.reshape((1,-1,1))
spot=np.atleast_3d(spot)
x_train=spot[:,:-1,:]/max_spot
y_train=spot[:,1:,:]/max_spot

#create net
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(units=20,return_sequences=True))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))

#train
model.compile(loss='mse', optimizer='adam')
history=model.fit(x_train, y_train, epochs=50)
loss_train=history.history["loss"]
model.summary()

# plot results
plt.plot(loss_train)
plt.title('mse loss')
plt.grid()

y_pred_tf = model.predict(x_train)

#save for inspection
TFUtils.save_tf_to_json(model,'saved_model_simplernn.json')

#reload for BeeDNN
model_beednn=BeeDNNUtils.load_from_json('saved_model_simplernn.json')
y_pred_beednn=model_beednn.predict(x_train)

plt.figure()
plt.plot(y_train.flatten()*max_spot,label='y_train')
plt.plot(y_pred_tf.flatten()*max_spot,label='y_pred_TF')
plt.plot(y_pred_beednn.flatten()*max_spot,label='y_pred_BeeDNN')
plt.legend()
plt.title('truth vs. pred')
plt.show()