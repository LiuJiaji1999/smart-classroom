import numpy as np
from sklearn import model_selection as mo
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
 
def iris_type(s):
    # 数据转为整型，数据集标签类别由string转为int
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]
 
data = np.loadtxt(r'D:/alpha/iris.data', dtype=float, delimiter=',', converters={4:iris_type})
'''
def loadtxt(fname, dtype=float, comments='#', delimiter=None,
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding='bytes', max_rows=None):
'''
x, y = np.split(data, (4, ), axis=1)
x_train, x_test, y_train, y_test = mo.train_test_split(x, y, random_state=1, test_size=0.3)
'''
train_data：被划分的样本特征集
train_target：被划分的样本标签
test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量
random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
'''
clf = svm.SVC(C=0.5, kernel='rbf', decision_function_shape='ovr')
clf.fit(x_train, y_train, sample_weight=None)
print(x_train.shape)
#print(x_train)
#print(y_train)
#print(x_test)
#print(y_test)
acc = clf.predict(x_train) == y_train.flat
print('Accuracy:%f' % (np.mean(acc)))
x1 = x[:, :2]
x_train, x_test, y_train, y_test = mo.train_test_split(x1, y,random_state=1, test_size=0.3)
clf.fit(x_train, y_train, sample_weight=None)
x1_min, x1_max = x1[:, 0].min(), x1[:, 0].max()
x2_min, x2_max = x1[:, 1].min(), x1[:, 1].max()
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
g_test = np.stack((x1.flat, x2.flat), axis=1)
print(g_test.shape)
g_map = clf.predict(g_test).reshape(x1.shape)
y = clf.predict(x_test)
cm_light = colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dack = colors.ListedColormap(['r', 'g', 'b'])
plt.pcolormesh(x1, x2, g_map, cmap=cm_light)
plt.scatter(x_test[:, 0], x_test[:, 1],c=np.squeeze(y.flat), s=50, cmap=cm_dack)
plt.plot()
plt.grid()
plt.show()
 