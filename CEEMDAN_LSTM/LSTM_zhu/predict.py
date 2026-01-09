import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn

# 1 数据处理 (# http://www.openit.cn/openData2/792.jhtml)
# road 1 61天
# 读取 CSV 文件
data = pd.read_csv(r"./road1.csv")


# 提取第 5 列数据 (索引从 0 开始)
# 注意：假设第 5 列不包含标签，可以直接用 data.iloc[:, 4] 提取
# 如果有标签列，通常需要忽略第一列，所以需要根据实际文件结构调整索引
fifth_column_data = data.iloc[:, 3][:144*31]
plt.figure(figsize=(15, 6))  # 宽度为 12，高度为 6
plt.plot(fifth_column_data)
plt.title('Line Plot of First 144 Data Points of 5th Column')
plt.xlabel('Index (First 144 Points)')
plt.ylabel('Value')

plt.xticks(range(0, 144*31, 144))  # 自定义 x 轴刻度（可选）
plt.show()

# 2 定义数据集
train_size = int(len(fifth_column_data)*(31/61))
test_size = len(fifth_column_data) - train_size
train, test = fifth_column_data[:train_size], fifth_column_data[train_size:]
train_tensor = torch.FloatTensor(train).view(-1, train.shape[0], 1)
test_tensor = torch.FloatTensor(test.values).view(-1, test.shape[0], 1)
# test_tensor = torch.FloatTensor(test).view(-1, test.shape[0], 1)

# 3 定义网络架构
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化h0和c0
        h0 = torch.rand(self.num_layers, x.size(0), self.hidden_size, requires_grad=True)
        c0 = torch.rand(self.num_layers, x.size(0), self.hidden_size, requires_grad=True)
        output, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(output[:, :, :])
        return out

if __name__ == "__main__":
    # 超参数设置
    input_size = 1
    hidden_size = 50
    num_layers = 1
    output_size = 1
    learning_rate = 0.01
    num_epochs = 300

    # 实例化模型
    model1 = LSTMModel(input_size, hidden_size, num_layers, output_size)
    # 加载模型
    model1.load_state_dict(torch.load('lstm_model.pth'))
    # 定义损失函数与优化算法
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    # 训练循环
    for epoch in range(num_epochs):
        outputs = model1(train_tensor)
        optimizer.zero_grad()
        loss = criterion(outputs, train_tensor[:, :, :])
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f'Epoch[{epoch+1}/{num_epochs}], Loss:{loss.item():.4f}')
            print("finish")


    # 5 预测测试集结果
    # 需要将测试集转换为适合模型输入的形状
    test_tensor = test_tensor.view(-1, test_tensor.shape[1], 1)

    # 前向传播以获得模型输出
    with torch.no_grad():
        model1.eval()  # 设置模型为评价模式
        test_outputs = model1(test_tensor)
        test_outputs = test_outputs.view(-1).numpy()  # 转换为numpy数组

    # 6 绘制结果
    # 转换测试集实际值为numpy数组
    actual_values = test.values

    # 绘制图像
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label='实际值', color='blue')
    plt.plot(test_outputs, label='期望值', color='red')
    plt.title('测试集的实际值与预测值对比')
    plt.xlabel('时间点')
    plt.ylabel('交通量')
    plt.legend()
    plt.show()

    # 假设 model1 是你的训练好的模型
    torch.save(model1.state_dict(), 'lstm_model.pth')
