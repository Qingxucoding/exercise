import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import time

def kmeans(X, num_clusters, initialization='random', max_iterations=300, convergence_tolerance=1e-4):
    # 第1步：初始化聚类中心
    if initialization == 'random':
        indices = np.random.choice(X.shape[0], num_clusters, replace=False)
        centroids = X[indices]
    elif initialization == 'kmeans++':
        centroids = kmeanspp(X, num_clusters)
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        # 第2步：为每个数据点分配最近的聚类中心
        # 计算每个点到所有聚类中心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        # 第3步：计算新的聚类中心
        new_centroids = np.zeros((num_clusters, X.shape[1]))
        for j in range(num_clusters):
            if np.any(labels == j):  # 如果存在 j 的样本
                new_centroids[j] = X[labels == j].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]  # 保持原中心不变
        # 检查收敛条件
        if np.all(np.abs(new_centroids - centroids) < convergence_tolerance):
            break
        centroids = new_centroids  # 更新聚类中心
    return centroids, labels, iterations

def kmeanspp(input, num_clusters):
    centroids = [input[np.random.choice(input.shape[0])]]  # 随机选择初始中心
    for _ in range(1, num_clusters):
        # 计算所有点到已选中心的最小距离
        squared_distances = np.min(np.linalg.norm(input[:, np.newaxis] - np.array(centroids), axis=2), axis=1)
        # 计算选择下一个中心的概率
        probabilities = squared_distances / squared_distances.sum()
        # 根据概率选择下一个中心
        new_centroid = input[np.random.choice(input.shape[0], p=probabilities)]
        centroids.append(new_centroid)
    return np.array(centroids)

def accuracy_computing(true_labels, predicted_labels):
    # 计算混淆矩阵
    num_classes = max(true_labels.max(), predicted_labels.max()) + 1
    cost_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            cost_matrix[i, j] = -np.sum((true_labels == i) & (predicted_labels == j))
    # 使用匈牙利算法解决标签重新分配问题
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    # 重新标记预测结果
    mapping = dict(zip(col_indices, row_indices))
    mapped_predicted_labels = np.array([mapping[label] for label in predicted_labels])
    # 计算准确率
    accuracy = accuracy_score(true_labels, mapped_predicted_labels)
    return accuracy

# 主程序入口
if __name__ == "__main__":
    # 加载数据
    train_set = pd.read_csv("mnist_train.csv").values
    test_set = pd.read_csv("mnist_test.csv").values

    # 预处理数据
    train_X = train_set[:, 1:] / 255.0
    train_y = train_set[:, 0]
    test_X = test_set[:, 1:] / 255.0
    test_y = test_set[:, 0]

    # 运行 K-Means 并测量时间
    start_time = time.time()
    centroids, cluster_labels, iteration_count = kmeans(test_X, 10, initialization='random')
    end_time = time.time()

    # 计算聚类准确率
    accuracy = accuracy_computing(test_y, cluster_labels)

    # 输出结果
    print("Initialization: random")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Iteration count: {iteration_count}")

    # 运行 K-Means++ 并测量时间
    start_time = time.time()
    centroids, cluster_labels, iteration_count = kmeans(test_X, 10, initialization='kmeans++')
    end_time = time.time()

    # 计算聚类准确率
    accuracy = accuracy_computing(test_y, cluster_labels)

    # 输出结果
    print("Initialization: kmeans++")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Iteration count: {iteration_count}")