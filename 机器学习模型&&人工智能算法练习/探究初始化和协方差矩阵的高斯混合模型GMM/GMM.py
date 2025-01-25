import numpy as np
from scipy import stats
import pandas as pd
import random
from sklearn.decomposition import PCA
import time

class GMM(object):
    def __init__(self, num_clusters, train_data_path, test_data_path, kmeans_mode, covars_mode):
        # 混合权重、均值和协方差矩阵
        self.weights = None
        self.means = None
        self.covars = None

        self.num_clusters = num_clusters  # 高斯分布个数
        self.max_weight = np.empty(self.num_clusters)  # 每个高斯分布中概率最大的一个点对应的标签，便于后序进行模型评估
        train_data = pd.read_csv(train_data_path).values  # 读取数据
        test_data = pd.read_csv(test_data_path).values
        # 归一化并降维
        self.train_labels = train_data[:, 0]
        self.train_features = train_data[:, 1:]/255.0
        self.test_labels = test_data[:, 0]
        self.test_features = test_data[:, 1:]/255.0
        pca = PCA(n_components=50)  # 使用PCA进行降维
        pca.fit(self.train_features)
        self.train_features = pca.transform(self.train_features)
        self.test_features = pca.transform(self.test_features)
        self.init(kmeans_mode, covars_mode)  # 初始化参数

    def init(self, mu_init_mode, sigma_init_mode):
        # 初始化权重
        self.weights = np.random.uniform(low=1e-7, high=1, size=self.num_clusters)
        self.weights = self.weights / self.weights.sum()  # 保证所有p_k的和为1
        # 初始化均值
        if mu_init_mode == 'random':
            self.means = self.random_initialization()
        elif mu_init_mode == '++':
            self.means = self.kmeanspp_initialization()
        # 初始化协方差矩阵
        self.covars = self.covars_initialization(sigma_init_mode)

    def random_initialization(self):
        """随机选择样本点作为均值"""
        random_ind = random.sample(range(len(self.train_features)), self.num_clusters)
        return self.train_features[random_ind]

    def kmeanspp_initialization(self):
        """K-Means++初始化方法"""
        # 随机选择第一个中心
        initial_index = np.random.choice(len(self.train_features), 1)[0]
        centroids = []
        centroids.append(self.train_features[initial_index])  # 将第一个选择的样本作为第一个中心

        # 循环来选择其他中心
        for _ in range(1, self.num_clusters):
            # 使用一个临时数组存储每个点到已选星心的最小距离
            min_distances = np.full(len(self.train_features), np.inf)  # 初始化为无穷大
            for c_idx in range(len(centroids)):
                centroid = centroids[c_idx]
                for sample_idx in range(len(self.train_features)):
                    sample = self.train_features[sample_idx]
                    # 计算样本与已选中心的平方距离
                    distance = np.sum(np.square(sample - centroid))
                    # 更新最小距离
                    if distance < min_distances[sample_idx]:
                        min_distances[sample_idx] = distance

            # 计算选择下一个中心的概率
            probabilities = min_distances / np.sum(min_distances)
            # 从样本中根据计算的概率选择下一个中心
            r = np.random.rand()  # 生成 0 到 1 之间的随机数
            cumulative_prob = 0.0
            next_index = None
            # 计算累积概率选择下一个中心
            for i in range(len(probabilities)):
                cumulative_prob += probabilities[i]
                if cumulative_prob >= r:  # 确定哪个样本被选择为下一个中心
                    next_index = i
                    break

            centroids.append(self.train_features[next_index])  # 添加新的中心到列表

        return np.array(centroids)  # 返回最终的中心点

    def covars_initialization(self, sigma_init_mode):
        """初始化协方差矩阵"""
        covars = np.empty((self.num_clusters, len(self.train_features[0]), len(self.train_features[0])))
        for i in range(self.num_clusters):
            if sigma_init_mode == 'diag_equal':
                covars[i] = np.eye(len(self.train_features[0])) * (random.random() + 1e-8)
            elif sigma_init_mode == 'diag_not_equal':
                covars[i] = np.diag(np.random.rand(len(self.train_features[0])) + 1e-8) * 0.8
            elif sigma_init_mode == 'random':
                # 使用实际数据协方差矩阵进行初始化
                covars[i] = 0.1 * np.cov(self.train_features.T)
            covars[i] += np.eye(len(self.train_features[0])) * 1e-7
        return covars
    def train(self):
        epoch = 0
        while True:
            epoch += 1
            # print(f"Epoch {epoch}: Training...")  # 输出当前的 epoch 信息
            old_means = self.means
            old_weights = self.weights
            old_covars = self.covars
            # E步
            gamma, phi = self.e_step()
            # M步
            self.m_step(gamma)
            # 检查收敛
            if (np.allclose(old_weights, self.weights, atol=1e-4)
                    and np.allclose(old_means, self.means, atol=1e-4)
                    and np.allclose(old_covars, self.covars, atol=1e-4)):
                print(f"Converged after epoch {epoch}.")  # 收敛信息
                max_ind = np.argmax(gamma, axis=0)
                self.max_weight = self.train_labels[max_ind]
                break

    def e_step(self):
        phi = np.zeros((self.train_features.shape[0], self.num_clusters))
        for i in range(self.num_clusters):
            # 检查协方差是否有效，避免NaN或inf
            if np.isnan(self.covars[i]).any() or np.isinf(self.covars[i]).any():
                raise ValueError(f"Covariance matrix {i} contains NaN or inf.")

            # 生成 K 个概率密度函数并计算所有样本的概率密度
            a = self.weights[i] * stats.multivariate_normal.pdf(self.train_features, mean=self.means[i],
                                                                cov=self.covars[i]) + 1e-10
            phi[:, i] = a

        total_prob = phi.sum(axis=1).reshape(-1, 1)

        # 检查生成的概率密度，为避免除以0
        if np.any(total_prob == 0):
            raise ValueError("Total probability for a sample is zero, cannot normalize.")

        gamma = phi / total_prob  # 计算后验概率
        return gamma, phi

    def m_step(self, gamma):
        weights_hat = gamma.sum(axis=0)  # 目前是phi_ij对i求和
        means_hat = np.tensordot(gamma, self.train_features, axes=[0, 0]) / weights_hat.reshape(-1,
                                                                                       1)  # 表示让gamma的第0维(列)和x的第0维(列)作内积
        covars_hat = np.empty(self.covars.shape)
        for i in range(self.num_clusters):
            tmp = self.train_features - self.means[i]
            covars_hat[i] = np.dot(tmp.T * gamma[:, i], tmp)
            covars_hat[i] /= weights_hat[i]
        # 更新参数
        self.covars = covars_hat + np.eye(len(self.train_features[0])) * 1e-7
        self.means = means_hat
        weights_hat = weights_hat / len(self.train_features)
        self.weights = weights_hat

    def test(self):
        acc = 0
        for i, x in enumerate(self.test_features):  # 遍历每个测试样本，计算准确率
            p = [-1 for _ in range(self.num_clusters)]
            for j in range(self.num_clusters):
                p[j] = stats.multivariate_normal.pdf(x, self.means[j], self.covars[j])
            owner = self.max_weight[np.argmax(p)]
            if owner == self.test_labels[i]:
                acc += 1
        acc = acc / len(self.test_labels) * 100
        print("Accuracy: %.2f %%" % acc)
        return acc

if __name__ == '__main__':
    # 定义初始化模式和协方差矩阵的类型
    kmeans_modes = ['random', '++']
    covars_modes = ['diag_equal', 'diag_not_equal', 'random']

    # 遍历每种 K-Means 初始化模式
    for kmeans_mode in kmeans_modes:
        #遍历每种协方差矩阵类型
        for covars_mode in covars_modes:
            # 初始化 GMM 模型
            gmm = GMM(10, 'mnist_train.csv', 'mnist_test.csv', kmeans_mode, covars_mode)
            print(f"K-Means Initialization Mode: {kmeans_mode}")
            print(f"Covariance Initialization Mode: {covars_mode}")

            gmm.init(kmeans_mode, covars_mode)
            t1 = time.time()
            gmm.train()
            t2 = time.time()
            gmm.test()
            print("time: %.3f s = %.3f mins" % ((t2 - t1), (t2 - t1) / 60))

            print("\n")  # 在不同配置组之间插入换行


