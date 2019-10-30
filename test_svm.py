import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

model=joblib.load("svm_model.m")
print()
data_0 = np.loadtxt(".\\gesture_data\\quan\\quan_1.txt")
data_1 = np.loadtxt(".\\gesture_data\\zhua\\zhua_1.txt")
data_2 = np.loadtxt(".\\gesture_data\\zhang\\zhang_1.txt")
data_3 = np.loadtxt(".\\gesture_data\\zhi\\zhi_1.txt")
data_4 = np.loadtxt(".\\gesture_data\\zan\\zan_1.txt")
data_5 = np.loadtxt(".\\gesture_data\\qiang\\qiang_1.txt")
data_6 = np.loadtxt(".\\gesture_data\\ok\\ok_1.txt")

data_x_0 = np.vstack((data_0, data_1))
data_x_1 = np.vstack((data_2, data_3))
data_x_2 = np.vstack((data_4, data_5))
data_x_a = np.vstack((data_x_0, data_x_1))
data_x_b = np.vstack((data_x_2, data_6))
data_x = np.vstack((data_x_a, data_x_b))

a_list = np.zeros(645)
b_list = np.ones(573)
c_list = np.ones(1003) * 2
d_list = np.ones(849) * 3
e_list = np.ones(1459) * 4
f_list = np.ones(1500) * 5
g_list = np.ones(2236) * 6

data_y_0 = np.append(a_list, b_list)
data_y_1 = np.append(c_list, d_list)
data_y_2 = np.append(e_list, f_list)
data_y_a = np.append(data_y_0, data_y_1)
data_y_b = np.append(data_y_2, g_list)
data_y = np.append(data_y_a, data_y_b)

index = [i for i in range(len(data_x))]
np.random.shuffle(index)  # 打乱数据
data_x = data_x[index]
data_y = data_y[index]
x = data_x
y = data_y
scaler = StandardScaler()
x_std = scaler.fit_transform(x)  # 标准化
score = model.score(x_std , data_y)
print(score)