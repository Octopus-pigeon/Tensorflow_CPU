# coding= utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

d_x=np.arange(50)
d_c=np.random.random(50)
d_y=-5*d_x+d_c*10
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
print(model.summary())
model.compile(optimizer='adam',loss='mse')
model.fit(d_x,d_y,epochs=50)
y=model.predict(d_x)
# print(d_x)
# print(y)
plt.figure()
plt.plot(d_x, d_y,'r')
plt.plot(d_x, y,'b')
plt.show()





