import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# 生成正弦波样本
x = np.linspace(0, 2 * np.pi, 100)  # 从0到2π生成100个点
y = np.sin(x)  # 标准正弦曲线

# 添加白噪声
noise = np.random.normal(0, 0.1, y.shape)  # 均值为0，标准差为0.1的高斯噪声
y_noisy = y + noise  # 带有噪声的正弦数据

# 数据准备
X = x.reshape(-1, 1)  # 转换为列向量
Y = y_noisy.reshape(-1, 1)

# 转换为 PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# 手动拆分训练集和测试集
np.random.seed(42)
indices = np.random.permutation(len(X_tensor))
split_index = int(len(X_tensor) * 0.8)  # 80%为训练集
train_indices, test_indices = indices[:split_index], indices[split_index:]

X_train = X_tensor[train_indices]
Y_train = Y_tensor[train_indices]
X_test = X_tensor[test_indices]
Y_test = Y_tensor[test_indices]

# 定义 BP 神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden1 = nn.Linear(1, 20)  # 输入层到第一个隐藏层
        self.hidden2 = nn.Linear(20, 20)  # 第一个到第二个隐藏层
        self.output = nn.Linear(20, 1)  # 第二个隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.hidden1(x))  # ReLU 激活函数
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

# 实例化模型、定义损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    optimizer.zero_grad()  # 清零梯度
    outputs = model(X_train)  # 前向传播
    loss = criterion(outputs, Y_train)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if (epoch + 1) % 100 == 0:  # 每100轮打印一次损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 使用训练后的模型进行预测
model.eval()  # 设置为评估模式
with torch.no_grad():
    Y_pred = model(X_tensor)

# 设置 Matplotlib 使用的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
plt.rcParams['axes.unicode_minus'] = False  # 防止负号的乱码


# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y_noisy, color='red', label='带白噪声的正弦样本数据')
plt.plot(X, y, color='blue', label='标准正弦曲线')
plt.plot(X, Y_pred.numpy(), color='green', label='BP神经网络拟合曲线')
plt.title('BP神经网络拟合带有白噪声的正弦数据')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
