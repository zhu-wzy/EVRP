import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 创建数据字典并转换为 DataFrame
data = {
    "年份": list(range(1990, 2010)),
    "公路客运量/万人": [
        5126, 6217, 7730, 9145, 10460, 11387, 12353,
        15750, 18304, 19836, 21024, 19490, 20433,
        22598, 25107, 33442, 36836, 40548, 42927, 43462
    ],
    "公路货运量/万吨": [
        1237, 1379, 1385, 1399, 1663, 1714, 1834,
        4322, 8132, 8936, 11099, 11203, 10524,
        11115, 13320, 16762, 18673, 20724, 20803, 21804
    ]
}

df = pd.DataFrame(data)

# 手动进行 Min-Max 标准化
def min_max_normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)

features = df[["公路客运量/万人", "公路货运量/万吨"]].values
features_scaled = min_max_normalize(features)

# 划分输入和输出
X = features_scaled[:-1]  # 1990-2009年的数据
y = features_scaled[1:]   # 1991-2010年的数据

# 转换为 PyTorch 的张量
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

class BPNetwork(nn.Module):
    def __init__(self):
        super(BPNetwork, self).__init__()
        self.hidden1 = nn.Linear(2, 10)  # 输入层到隐藏层1
        self.hidden2 = nn.Linear(10, 10)  # 隐藏层1到隐藏层2
        self.output = nn.Linear(10, 2)    # 隐藏层2到输出层

    def forward(self, x):
        x = torch.relu(self.hidden1(x))   # 使用 ReLU 激活函数
        x = torch.relu(self.hidden2(x))
        x = self.output(x)                 # 输出层
        return x

# 实例化模型
model = BPNetwork()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 定义训练的 epoch
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()       # 训练模式
    optimizer.zero_grad()  # 清空梯度

    # 前向传播
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)  # 计算损失

    # 反向传播
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 进行预测
model.eval()  # 评估模式
with torch.no_grad():
    last_year_data = torch.FloatTensor(features_scaled[-1:])  # 取2009年的数据进行预测
    predictions = model(last_year_data)

# 反向标准化
def inverse_min_max_normalize(data, original_data):
    min_val = np.min(original_data, axis=0)
    max_val = np.max(original_data, axis=0)
    return data * (max_val - min_val) + min_val

# 将预测结果从标准化的值转换回原始值
predicted_values = inverse_min_max_normalize(predictions.numpy(), features)

# 输出2010年和2011年的预测值
print("2010年预测公路客运/万人:", predicted_values[0][0])
print("2010年预测公路货运/万吨:", predicted_values[0][1])



# 历史数据
years = list(range(1990, 2012))
historical_passenger = [
    5126, 6217, 7730, 9145, 10460, 11387, 12353,
    15750, 18304, 19836, 21024, 19490, 20433,
    22598, 25107, 33442, 36836, 40548, 42927, 43462
]
historical_freight = [
    1237, 1379, 1385, 1399, 1663, 1714, 1834,
    4322, 8132, 8936, 11099, 11203, 10524,
    11115, 13320, 16762, 18673, 20724, 20803, 21804
]

# 预测数据
predicted_passenger = [predicted_values[0][0]]  # 2010年预测
predicted_freight = [predicted_values[0][1]]    # 2010年预测

# 绘制图形
plt.figure(figsize=(14, 6))
# 设置 Matplotlib 使用的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
plt.rcParams['axes.unicode_minus'] = False  # 防止负号的乱码

# 公路客运量
plt.subplot(1, 2, 1)
plt.plot(years[:-2], historical_passenger, marker='o', label='历史公路客运量', color='blue')
plt.scatter(2010, predicted_passenger[0], color='red', label='2010年预测公路客运量')
plt.title('公路客运量预测')
plt.xlabel('年份')
plt.ylabel('公路客运量（万人）')
plt.legend()
plt.grid()

# 公路货运量
plt.subplot(1, 2, 2)
plt.plot(years[:-2], historical_freight, marker='o', label='历史公路货运量', color='orange')
plt.scatter(2010, predicted_freight[0], color='red', label='2010年预测公路货运量')
plt.title('公路货运量预测')
plt.xlabel('年份')
plt.ylabel('公路货运量（万吨）')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
