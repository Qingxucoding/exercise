import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 修改卷积层数量和滤波器数量的模型
class LeNet(nn.Module):
    def __init__(self, num_conv_layers=2, num_filters=16):
        super(LeNet, self).__init__()
        # 存放不同的卷积层
        self.convs = nn.ModuleList()
        in_channels = 3
        height, width = 32,32

        # 定义不同卷积层层的类型
        for _ in range(num_conv_layers):
            self.convs.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1))
            in_channels = num_filters
            num_filters *= 2
            # 每次卷积后，图像的高度和宽度减半
            height = height // 2
            width = width // 2

        # 定义一个最大池化层，池化核大小为2x2，步长为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        flat_features = in_channels * height * width
        self.fc1 = nn.Linear(flat_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
            x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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


layers = 2
filters = 64
print("====================================")
print(f"layers:{layers}, filters: {filters}")
modified_lenet = LeNet(layers, filters)
train_losses, test_accuracy, train_time = train_and_test_model(modified_lenet, X_train, Y_train, X_test, Y_test, epochs=30, lr=0.001)
print(f"Accuracy: {test_accuracy:.2f}, Training time: {train_time:.2f}s")
