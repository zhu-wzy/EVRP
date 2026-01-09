import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
# 读取CSV文件
data1 = pd.read_csv('交通流量预测结果.csv')  # 修改为您的实际文件路径
data2 = pd.read_csv('实际结果.csv')  # 修改为您的实际文件路径

# 提取需要绘制的数据
y1 = data1['SUM_Predicted']  # 从file1.csv提取B列
y2 = data2['SUM_Actual']  # 从file2.csv提取D列
x = range(len(y2))  # 假设x轴为简单的索引；可以根据实际需要调整

# 绘制数据
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(x, y1, label='预测数据', color='blue')  # 绘制File 1的数据
plt.plot(x, y2, label='实际数据', color='orange')  # 绘制File 2的数据

# 添加标注

plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.ylim(bottom=20)
# 显示图形
plt.show()
