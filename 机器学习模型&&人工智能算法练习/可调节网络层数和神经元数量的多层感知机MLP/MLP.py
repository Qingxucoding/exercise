import torch.optim as optim
import time
import torch
import torch.nn as nn

# 定义 MLP 模型类
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_neurons, output_size):
        super(MLP, self).__init__()
        # 初始化层列表
        self.layers = nn.ModuleList()

        # 添加输入层到第一个隐藏层的线性层和ReLU激活函数
        self.layers.append(nn.Linear(input_size, hidden_neurons))
        # 激活函数，确保非线性分类器
        self.layers.append(nn.ReLU(inplace=True))

        # 添加中间的隐藏层
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            self.layers.append(nn.ReLU(inplace=True))

        # 添加最后一个隐藏层到输出层的线性层
        self.layers.append(nn.Linear(hidden_neurons, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
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

# 数据预处理并转为tensor
def pre_processing(X_train, Y_train, X_test, Y_test):
    # X 转换为 tensor并归一化
    X_train_normalized = torch.tensor(X_train / 255.0, dtype=torch.float)
    X_test_normalized = torch.tensor(X_test / 255.0, dtype=torch.float)
    Y_train=torch.tensor(Y_train,dtype=int)
    Y_test=torch.tensor(Y_test,dtype=int)
    return X_train_normalized, Y_train, X_test_normalized, Y_test

# 根据传入的优化器类型返回相应的 PyTorch 优化器实例
def select_optimizer(optimizer_type, model):
    # 定义使用的优化器类型
    optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=0.01),
        'SGD_Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        'Adam': optim.Adam(model.parameters(), lr=0.001)
    }
    # 如果提供的优化器类型不在字典中，会抛出错误
    if optimizer_type not in optimizers:
        raise ValueError("Unsupported optimizer type!")
    return optimizers[optimizer_type]

def train_model(model, criterion, optimizer, X_train, Y_train, epochs):
    # 将模型设置为训练模式
    model.train()
    # 在每个轮次中，清零梯度，进行前向传播以计算输出，计算损失，反向传播梯度并更新模型参数
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

def evaluate_model(model, X_test, Y_test):
    # 将模型设置为评估模式
    model.eval()
    # 进行前向传播计算输出，找出最大概率的预测类别，并计算正确分类的样本数量
    # 评估模式下，模型只需要做前向传播，而不需要计算梯度来更新权重
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        total = Y_test.size(0)
        correct = (predicted == Y_test).sum().item()
    return 100 * correct / total

# 训练评估
def train_and_evaluate(X_train, Y_train, X_test, Y_test, layers, neurons, optimizers, epochs=30):
    print(f"Training with {layers} hidden layers, {neurons} neurons, Optimizer: {optimizers}")
    # 将数据移动到同一设备上
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    # 模型初始化
    input_size = 3072
    output_size = 10
    model = MLP(input_size, layers, neurons, output_size).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = select_optimizer(optimizers, model)

    # 训练模型
    start_time = time.time()
    train_model(model, criterion, optimizer, X_train, Y_train, epochs)
    end_time = time.time()

    # 测试模型
    accuracy = evaluate_model(model, X_test, Y_test)
    print(f"Accuracy: {accuracy:.2f}%, Training time: {end_time - start_time:.2f}s")
    print("===================================================")


# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, Y_train, X_test, Y_test=load_data('data')
    X_train, Y_train, X_test, Y_test=pre_processing(X_train, Y_train, X_test, Y_test)

    # 分不同参数的训练,layer = 1/2/3, neuron = 16/32/64, optimizer = 'SGD', 'SGD_Momentum', 'Adam'
    # 比较相同模型不同层数
    print("-----------different layers-----------")
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 1, 64, 'SGD')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 2, 64, 'SGD')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 64, 'SGD')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 1, 64, 'SGD_Momentum')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 2, 64, 'SGD_Momentum')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 64, 'SGD_Momentum')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 1, 64,  'Adam')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 2, 64,  'Adam')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 64,  'Adam')

    # 比较相同模型不同神经元数量
    print("-----------different neurons-----------")
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 16, 'SGD')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 32, 'SGD')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 64, 'SGD')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 16, 'SGD_Momentum')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 32, 'SGD_Momentum')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 64, 'SGD_Momentum')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 16,  'Adam')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 32,  'Adam')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 64,  'Adam')

    # 比较不同模型
    print("-----------different optimizers-----------")
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 64, 'SGD')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 64, 'SGD_Momentum')
    train_and_evaluate(X_train, Y_train, X_test, Y_test, 3, 64, 'Adam')
