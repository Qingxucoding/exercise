import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


# 重写后的hinge损失函数
def modified_hinge_loss(predictions, labels):
    return torch.mean(torch.max(torch.tensor(0.0), 1 - labels * predictions))


# 重写后的交叉熵损失函数
def modified_cross_entropy_loss(logits, binary_targets):
    eps = 1e-10
    probabilities = torch.sigmoid(logits)
    term1 = binary_targets * torch.log((probabilities + eps).clamp(max=1))
    term2 = (1 - binary_targets) * torch.log((1 - probabilities + eps).clamp(max=1))
    return -torch.mean(term1 + term2)


# 封装数据加载和预处理函数
def load_and_preprocess_data(train_csv_file, test_csv_file, use_cross_entropy=False):
    # 读取训练集和测试集数据
    train_set = pd.read_csv(train_csv_file).values
    test_set = pd.read_csv(test_csv_file).values

    # 分离特征和标签
    train_X = train_set[:, 1:]
    train_y = train_set[:, 0]
    test_X = test_set[:, 1:]
    test_y = test_set[:, 0]

    # 归一化到[0, 1]
    train_X = train_X.astype('float32') / 255.0
    test_X = test_X.astype('float32') / 255.0

    # 计算训练集的均值和标准差
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)

    # 避免除以零
    std[std == 0] = 1

    # 标准化，对于标准差为零的特征不进行标准化
    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    # 将NumPy数组转换为PyTorch张量
    train_X_tensor = torch.from_numpy(train_X).float()
    test_X_tensor = torch.from_numpy(test_X).float()

    # 根据损失函数类型预处理标签
    if use_cross_entropy:
        train_y_tensor = torch.from_numpy(train_y).float()
        test_y_tensor = torch.from_numpy(test_y).float()
    else:
        train_y_tensor = torch.from_numpy(train_y).view(-1, 1).float() * 2 - 1
        test_y_tensor = torch.from_numpy(test_y).view(-1, 1).float() * 2 - 1

    # 创建TensorDataset和DataLoader
    train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(test_X_tensor, test_y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


class Linear_classifier:
    def __init__(self, entropy, train_loader, test_loader):
        self.entropy = entropy
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 定义模型和优化器
        self.model = torch.nn.Linear(self.train_loader.dataset.tensors[0].shape[1], 1)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    def train(self):
        # 训练模型
        epochs = 1000
        for epoch in range(epochs):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                if self.entropy == 1:
                    loss = modified_cross_entropy_loss(output, target)
                else:
                    loss = modified_hinge_loss(output, target)
                loss.backward()
                self.optimizer.step()
            if (epoch + 1) % 100 == 0:  # 每50次训练打印评估参数
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def test(self):
        # 测试模型
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                test_outputs = self.model(data)
                predictions = (test_outputs >= 0).float() * 2 - 1
                total_correct += (predictions.round() == target).sum().item()
        accuracy = total_correct / len(self.test_loader.dataset)
        print(f'Total correct: {total_correct}')
        print(f'Total samples: {len(self.test_loader.dataset)}')
        if(self.entropy == False): accuracy = accuracy * 100
        print(f'Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    train_csv_file = "mnist_01_train.csv"
    test_csv_file = "mnist_01_test.csv"
    use_cross_entropy = False # 设置为True以使用交叉熵损失，否则使用hinge损失
    train_loader, test_loader = load_and_preprocess_data(train_csv_file, test_csv_file, use_cross_entropy)
    classifier = Linear_classifier(use_cross_entropy, train_loader, test_loader)
    classifier.train()
    classifier.test()