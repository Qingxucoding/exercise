import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 基础 LeNet 模型
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
#         self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一个卷积层：输入通道3，输出通道16，核大小5
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)  # padding保持特征图大小
        # 第二个卷积层：输入通道16，输出通道16，核大小5
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        # 计算全连接层输入特征数：假设输入图像大小为32x32
        # 经过两次卷积后，特征图大小变为 (16, 32, 32)
        self.fc1 = nn.Linear(16 * 32 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 应用第一个卷积层和ReLU激活函数
        x = F.relu(self.conv1(x))
        # 应用第二个卷积层和ReLU激活函数
        x = F.relu(self.conv2(x))
        # 展平特征图，准备输入到全连接层
        x = torch.flatten(x, 1)
        # 应用第一个全连接层和ReLU激活函数
        x = F.relu(self.fc1(x))
        # 应用第二个全连接层和ReLU激活函数
        x = F.relu(self.fc2(x))
        # 应用第三个全连接层（输出层）
        x = self.fc3(x)
        return x
# 数据导入
def load_data(dir):
    import pickle
    import numpy as np
    X_train = []
    Y_train = []
    for i in range(1, 6):
        with open(dir + r'/data_batch_' + str(i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        X_train.append(dict[b'data'])
        Y_train += dict[b'labels']
    X_train = np.concatenate(X_train, axis=0)
    with open(dir + r'/test_batch', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X_test = dict[b'data']
    Y_test = dict[b'labels']
    return X_train, Y_train, X_test, Y_test

# 数据预处理，并且全部转为 PyTorch Tensor
def pre_processing(X_train, Y_train, X_test, Y_test):
    # 从一维数组转为四维数据（样本数, 通道数, 高度, 宽度）并从 [0, 255] 范围缩放到 [0, 1] 范围
    X_train = X_train.reshape(-1, 3, 32, 32) / 255.0
    X_test = X_test.reshape(-1, 3, 32, 32) / 255.0
    # 转为pytorch张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
    return X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor

def get_data_loaders(X_train, Y_train, X_test, Y_test, batch_size=64):
    # 将数据集打包为训练集和测试集，便于后序迭代
    # 其中shuffle表示是否随机打乱数据
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test, Y_test),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    # 开启训练模式
    model.train()
    running_loss = 0.0
    # 迭代训练集中的数据
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_and_test_model(model, X_train, Y_train, X_test, Y_test, epochs=1000, lr=0.001, batch_size=64):
    model = model.to(device)
    # 初始化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = get_data_loaders(X_train, Y_train, X_test, Y_test, batch_size)
    start_time = time.time()
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_accuracy = evaluate_model(model, test_loader, device)
        if (epoch + 1)%10==0 :
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    end_time = time.time()
    return test_accuracy, end_time - start_time

X_train, Y_train, X_test, Y_test=load_data('data')
X_train, Y_train, X_test, Y_test=pre_processing(X_train, Y_train, X_test, Y_test)

lenet = LeNet()
print("====================================")
print("layers:{2}, filters: {16}")
train_losses, test_accuracy, train_time = train_and_test_model(lenet, X_train, Y_train, X_test, Y_test, epochs=100, lr=0.001)
print(f"Accuracy: {test_accuracy:.2f}, Training time: {train_time:.2f}s")
