

# from pandas import DataFrame
# df = DataFrame()
# df['t'] = [x for x in range(10)]
# df['t-1'] = df['t'].shift(-1)
# print(df)

from pandas import DataFrame
from pandas import concat
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

data_0 = np.loadtxt(".\\gesture_data\\quan\\quan_0.txt")
data_1 = np.loadtxt(".\\gesture_data\\zhua\\zhua_0.txt")
data_2 = np.loadtxt(".\\gesture_data\\zhang\\zhang_0.txt")
data_3 = np.loadtxt(".\\gesture_data\\zhi\\zhi_0.txt")
data_4 = np.loadtxt(".\\gesture_data\\zan\\zan_0.txt")
data_5 = np.loadtxt(".\\gesture_data\\qiang\\qiang_0.txt")
data_6 = np.loadtxt(".\\gesture_data\\ok\\ok_0.txt")

data_x_0 = np.vstack((data_0, data_1))
data_x_1 = np.vstack((data_2, data_3))
data_x_2 = np.vstack((data_4, data_5))
data_x_a = np.vstack((data_x_0, data_x_1))
data_x_b = np.vstack((data_x_2, data_6))
train_data_x = np.vstack((data_x_a, data_x_b))

a_list = np.zeros(554)
b_list = np.ones(788)
c_list = np.ones(983) * 2
d_list = np.ones(963) * 3
e_list = np.ones(1303) * 4
f_list = np.ones(1715) * 5
g_list = np.ones(1218) * 6
data_y_0 = np.append(a_list, b_list)
data_y_1 = np.append(c_list, d_list)
data_y_2 = np.append(e_list, f_list)
data_y_a = np.append(data_y_0, data_y_1)
data_y_b = np.append(data_y_2, g_list)
train_data_y = np.append(data_y_a, data_y_b)


# train_data_0 = np.loadtxt(".\\gesture_data\\quan\\quan_0.txt")
# train_data_1 = np.loadtxt(".\\gesture_data\\zhua\\zhua_0.txt")
# train_data_2= np.loadtxt(".\\gesture_data\\zhang\\zhang_0.txt")
# train_data_3 = np.loadtxt(".\\gesture_data\\zhi\\zhi_0.txt")
#
#
# print(train_data_0.shape,train_data_1.shape,train_data_2.shape,train_data_3.shape,)
# train_data_x = np.vstack((np.vstack((train_data_0, train_data_1)),np.vstack((train_data_2, train_data_3))))
# print( train_data_x.shape )
#
# train_data_y = np.vstack((np.vstack((np.zeros([554,1]),np.ones([788,1]))),np.vstack((np.ones([983,1])*2,np.ones([963,1])*3))))

# print(train_data_y[:5],train_data_y[-5:],train_data_y.shape)

index = [i for i in range(len(train_data_x))]
np.random.shuffle(index)
train_data_x= train_data_x[index]
train_data_y= train_data_y[index]

input_x=train_data_x[:6000,]
input_y=train_data_y[:6000,]
# print(train_data_x[:10,])
# print(train_data_y[:10,])
#
# test_data_0 = np.loadtxt(".\\gesture_data\\quan\\quan_1.txt")
# tset_data_1 = np.loadtxt(".\\gesture_data\\zhua\\zhua_1.txt")
# # print(test_data_0.shape,tset_data_1.shape)
# test_data_x = np.vstack((test_data_0, tset_data_1))
# # print(test_data_x.shape)
# test_data_y = np.vstack((np.zeros([645,1]),np.ones([573,1])))
# # print(test_data_y[:5],test_data_y[-5:],test_data_y.shape)
# index_t = [i for i in range(len(test_data_x))]
# np.random.shuffle(index_t)
test_x=train_data_x[6000:,]
test_y=train_data_y[6000:,]

inputs = keras.Input(shape=(5,), name='mnist_input')
h1 = keras.layers.Dense(64, activation='relu')(inputs)
h1 = keras.layers.Dense(64, activation='relu')(h1)
outputs = keras.layers.Dense(7, activation='softmax')(h1)
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam',
             loss=keras.losses.SparseCategoricalCrossentropy(),
             metrics=[keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(input_x , input_y, batch_size=100, epochs=100)
print('history:')
print(history.history)

model.save('ann_model.h5')
x=range(100)
y=history.history['loss']
z=history.history['sparse_categorical_accuracy']
# print(x,y)

result = model.evaluate(test_x, test_y)
print(result)

ss=model.predict(train_data_x[-10:,])
print(train_data_y[-10:,])
print(ss.argmax(axis=1))

plt.figure()
plt.plot(x,y,'r')
plt.plot(x,z,'b')
plt.show()


#
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     """
#     Frame a time series as a supervised learning dataset.
#     Arguments:
#         data: Sequence of observations as a list or NumPy array.
#         n_in: Number of lag observations as input (X).
#         n_out: Number of observations as output (y).
#         dropnan: Boolean whether or not to drop rows with NaN values.
#     Returns:
#         Pandas DataFrame of series framed for supervised learning.
#     """
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
#     # put it all together
#     agg = concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg
# values = [x for x in range(10)]
# data = series_to_supervised(values,2,2)
# print(data)
