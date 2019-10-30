#coding= utf-8
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

label=['quan','zhua','zhang','zhi','zan','qiang','ok']
def load_data(filename):
    '''
    假设这是鸢尾花数据,csv数据格式为：
    0,5.1,3.5,1.4,0.2
    0,5.5,3.6,1.3,0.5
    1,2.5,3.4,1.0,0.5
    1,2.8,3.2,1.1,0.2
    每一行数据第一个数字(0,1...)是标签,也即数据的类别。
    '''

    data_0 = np.loadtxt(".\\gesture_data\\quan\\quan_0.txt")
    data_1 = np.loadtxt(".\\gesture_data\\zhua\\zhua_0.txt")
    data_2 = np.loadtxt(".\\gesture_data\\zhang\\zhang_0.txt")
    data_3 = np.loadtxt(".\\gesture_data\\zhi\\zhi_0.txt")
    data_4 = np.loadtxt(".\\gesture_data\\zan\\zan_0.txt")
    data_5 = np.loadtxt(".\\gesture_data\\qiang\\qiang_0.txt")
    data_6 = np.loadtxt(".\\gesture_data\\ok\\ok_0.txt")

    data_x_0 = np.vstack((data_0, data_1))
    data_x_1 = np.vstack((data_2, data_3))
    data_x_2= np.vstack((data_4, data_5))
    data_x_a = np.vstack((data_x_0, data_x_1))
    data_x_b =np.vstack((data_x_2, data_6))
    data_x = np.vstack((data_x_a, data_x_b))

    # data_y = np.vstack((np.zeros([645, 1]), np.ones([573, 1])))
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
    data_y_a =np.append(data_y_0, data_y_1)
    data_y_b = np.append(data_y_2, g_list)
    data_y = np.append(data_y_a, data_y_b)

    index = [i for i in range(len(data_x))]
    np.random.shuffle(index)#打乱数据
    data_x = data_x[index]
    data_y = data_y[index]

    print("!!!",data_x.shape)
    # data = np.genfromtxt(filename, delimiter=',')
    # x = data[:, 1:]  # 数据特征
    # y = data[:, 0].astype(int)  # 标签
    x=data_x
    y=data_y
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)  # 标准化
    # 将数据划分为训练集和测试集，test_size=.3表示30%的测试集
    x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=.3)
    return x_train, x_test, y_train, y_test


def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced',)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)
    # 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train)
    # 计算测试集精度
    score = grid.score(x_test, y_test)
    y_pre=grid.predict(x_test)
    # print(pre)
    # print(y_test)
    print('最佳参数：',grid.best_params_)
    print('精度为%s' % score)
    joblib.dump(grid, "svm_model.m")
    print(classification_report(y_test, y_pre))

    mat=confusion_matrix(y_test,y_pre)
    print(mat)
    sns.heatmap(mat,square=True,annot=True,fmt='d',cbar=False,
                xticklabels=label,yticklabels=label)
    plt.xlabel('true label')
    plt.ylabel('prediced label')
    plt.show()

#
if __name__ == '__main__':
    svm_c(*load_data('example.csv'))

#导入svm和数据集
# from sklearn import svm,datasets
# #调用SVC()
# clf = svm.SVC()
# #载入鸢尾花数据集
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# print(X.shape,y.shape)
# #fit()训练
# clf.fit(X,y)
# #predict()预测
# pre_y = clf.predict(X[5:10])
# print(pre_y)
# print(y[5:10])
# #导入numpy
# import numpy as np
# test = np.array([[5.1,2.9,1.8,3.6]])
# #对test进行预测
# test_y = clf.predict(test)
# print(test_y)