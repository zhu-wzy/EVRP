import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 读取CSV文件
# data1 = pd.read_csv('交通流量预测结果.csv')  # 修改为您的实际文件路径
data2 = pd.read_csv('road1.csv')  # 修改为您的实际文件路径

# 提取需要绘制的数据
# y1 = data1['SUM_Predicted']  # 从file1.csv提取B列
y2 = data2['speed'][:4464]  # 从 file2.csv 提取D列，并仅选择前4464个数据
x = range(len(y2))  # x轴为简单的索引，基于y2的长度

# 绘制数据
plt.figure(figsize=(10, 6))  # 设置图形大小
# plt.plot(x, y1, label='预测数据', color='blue')  # 绘制File 1的数据
plt.plot(x, y2, label='实际数据', color='orange')  # 绘制File 2的数据

# 添加标注
plt.title('数据集')
plt.xlabel('样本')  # 替换为实际的X轴标签
plt.ylabel('车速')  # Y轴标签  
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格

# 显示图形
plt.show()
