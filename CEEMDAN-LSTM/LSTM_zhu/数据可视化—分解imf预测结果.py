import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 文件数量和类别
num_files = 10
files = [f'forecast_results_IMF{i}.xlsx' for i in range(1, num_files + 1)]


# 创建一个2x5的子图，10个文件分别放在其中
fig, axs = plt.subplots(5, 2, figsize=(12, 20))  # 5行2列
axs = axs.flatten()  # 将2D数组展平，方便迭代

# 遍历文件，读取数据并绘制
for i, file in enumerate(files):
    # 读取Excel文件，假设第一行为列名
    try:
        data = pd.read_excel(file)

        # 绘制折线图
        axs[i].plot(data.index, data['Predicted'], color='blue')
        axs[i].plot(data.index, data['Actual'], color='orange')

        # 图的标注

        axs[i].set_xlabel('样本')

        axs[i].legend()  # 显示图例
        axs[i].grid(True)  # 显示网格

    except Exception as e:
        print(f"Error reading {file}: {e}")

# 调整布局，使得子图间距更合理
plt.tight_layout()
plt.show()
