import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置字体为支持中文
matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号的显示问题

# 调整所有字体的大小
matplotlib.rcParams['font.size'] = 15 # 设置字体大小

# 读取CSV文件
file_path = 'CEEMDAN.csv'  # 请修改为你的CSV文件路径
data = pd.read_csv(file_path)

# 计算需要的行数
num_columns = len(data.columns) - 1  # 第一列是索引或不需要绘制的列
num_rows = (num_columns + 1) // 2  # 每两列一行

# 创建一个窗口，调整纵向和横向大小，让图变窄
plt.figure(figsize=(10, num_rows * 2))  # 调整宽度为10，保持高度为原来的1/2

# 遍历每一列，绘制图形
for i, column in enumerate(data.columns[1:], start=1):  # 从第2列开始
    plt.subplot(num_rows, 2, i)  # 2列的子图设定
    plt.plot(data[column], color='b', linewidth=1.5)  # 设置折线颜色和粗细
    plt.title(f'{column}', fontsize=14)  # 设置标题为列名，字体稍大

    plt.ylabel('车速', fontsize=12)  # 设置Y轴标签，字体稍大
    plt.grid(True)  # 显示网格

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.5)  # 控制纵向间距

# 调整布局
plt.tight_layout()
plt.show()
