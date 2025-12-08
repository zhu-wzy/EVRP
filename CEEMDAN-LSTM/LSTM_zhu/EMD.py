import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
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
fifth_column_data = data.iloc[:, 3][:144*31]
train_size = int(len(fifth_column_data)*(31/61))
train = fifth_column_data[:train_size]

plt.figure(figsize=(15, 6))  # 宽度为 12，高度为 6
plt.plot(fifth_column_data)
plt.title('Line Plot of First 144 Data Points of 5th Column')
plt.xlabel('Index (First 144 Points)')
plt.ylabel('Value')

# 生成模拟信号数据
t = 144*31




# 绘制原始信号和含噪声的信号
plt.figure(figsize=(10, 4))
plt.subplot(211)
plt.title('Original Signal')
plt.plot(t, fifth_column_data)



# 使用EMD对含噪声的信号进行分解
emd = EMD()
IMFs = emd(fifth_column_data)

# 绘制分解后的IMFs
plt.figure(figsize=(10, 6))
plt.subplot(len(IMFs) + 1, 1, 1)
plt.title('IMFs')
plt.plot(t, fifth_column_data, label='Original Signal')

for i, IMF in enumerate(IMFs):
    plt.subplot(len(IMFs) + 1, 1, i + 2)
    plt.plot(t, IMF)
    plt.title(f'IMF {i+1}')

plt.tight_layout()
plt.show()
