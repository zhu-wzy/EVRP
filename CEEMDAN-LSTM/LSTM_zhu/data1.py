import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

# 1 数据处理 (# http://www.openit.cn/openData2/792.jhtml)
# road 1 61天
# 读取 CSV 文件
data = pd.read_csv(r"./road_1.csv")
# file_path = 'your_file.csv'  # 替换为您的 CSV 文件路径
# data = pd.read_csv(file_path)

# 提取第 5 列数据 (索引从 0 开始)
# 注意：假设第 5 列不包含标签，可以直接用 data.iloc[:, 4] 提取
# 如果有标签列，通常需要忽略第一列，所以需要根据实际文件结构调整索引
fifth_column_data = data.iloc[:, 3][:144]
fifth_column_data2 = data.iloc[:, 3][144:288]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 绘制折线图
axs[0].plot(fifth_column_data)
axs[0].set_title('road_1 in day1 (first 144)')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Value')
axs[0].grid()  # 添加网格

# 绘制第二幅图
axs[1].plot(fifth_column_data2)
axs[1].set_title('road_1 in day1 (next 144)')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Value')
axs[1].grid()  # 添加网格

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

