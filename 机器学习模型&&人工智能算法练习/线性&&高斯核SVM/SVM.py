from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# 导入数据
train_set = pd.read_csv("mnist_01_train.csv").values
test_set = pd.read_csv("mnist_01_test.csv").values

# 构造训练集和测试集
train_X = train_set[:, 1:]
train_y = train_set[:, 0]
test_X = test_set[:, 1:]
test_y = test_set[:, 0]

# 数据预处理：标准化
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)

# 数据预处理：归一化
min_max_scaler = MinMaxScaler()
train_X_min_max_scaled = min_max_scaler.fit_transform(train_X_scaled)
test_X_min_max_scaled = min_max_scaler.transform(test_X_scaled)

# 初始化不同参数设置的SVM分类器
# 初始化高斯核SVM，设置参数C和gamma。
clf_svc_rbf = svm.SVC(C=1, kernel='rbf', gamma='auto')
# 初始化线性核的SVM
clf_svc_linear = svm.SVC(C=1, kernel='linear', random_state=0)

# 将所有分类器放入列表中
clfs = [clf_svc_rbf, clf_svc_linear]

# 对每个分类器进行训练和绘图
for clf, i in zip(clfs, range(len(clfs))):
    time_start = time.time()
    clf.fit(train_X_min_max_scaled, train_y)  # 训练分类器
    time_end = time.time()
    # 计算准确率
    accuracy = clf.score(test_X_min_max_scaled, test_y)
    print("test accuracy:\t%f" % accuracy)
    print("time :\t%f" % (time_end - time_start))