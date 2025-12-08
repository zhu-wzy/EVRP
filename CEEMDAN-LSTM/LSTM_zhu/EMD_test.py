import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
import pandas as pd

# 读取 CSV 文件
data = pd.read_csv(r"./road1.csv")
fifth_column_data = data.iloc[:, 3][:144*31]
train_size = int(len(fifth_column_data)*(31/61))
train = fifth_column_data[:train_size]

plt.figure(figsize=(15, 6))  # 宽度为 15，高度为 6
plt.plot(fifth_column_data)
plt.title('Line Plot of First 144 Data Points of 5th Column')
plt.xlabel('Index (First 144 Points)')
plt.ylabel('Value')

# 定义 t 数组
t = np.linspace(0, len(fifth_column_data)-1, len(fifth_column_data))

# 绘制原始信号和含噪声的信号
plt.figure(figsize=(10, 4))
plt.subplot(211)
plt.title('Original Signal')
plt.plot(t, fifth_column_data)

# 使用EMD对含噪声的信号进行分解
emd = EMD()
# 将 Pandas Series 转换为 NumPy array
IMFs = emd(fifth_column_data.to_numpy())

# 绘制分解后的IMFs
plt.figure(figsize=(12, 8))
plt.subplot(len(IMFs) + 1, 1, 1)
plt.title('IMFs')
plt.plot(t, fifth_column_data, label='Original Signal')

for i, IMF in enumerate(IMFs):
    plt.subplot(len(IMFs) + 1, 1, i + 2)
    plt.plot(t[:len(IMF)], IMF)  # 仅绘制 IMF 的相同长度
    plt.title(f'IMF {i+1}')
    print(i+1)
plt.tight_layout()
plt.show()
