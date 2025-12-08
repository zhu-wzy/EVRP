import pandas as pd

# 读取CSV文件
data = pd.read_csv('speed-data.csv')  # 请将 'data.csv' 替换为实际文件名

# 选择要归一化的列
column_to_normalize = 'speed'  # 替换为您需要归一化的列名

# 计算最小值和最大值
min_value = data[column_to_normalize].min()
max_value = data[column_to_normalize].max()

# 归一化到0到2
data[column_to_normalize + '_normalized'] = ((data[column_to_normalize] - min_value) / (max_value - min_value)) * 2

# 保存归一化后的数据到新的CSV文件
output_file = 'speed_normalized.csv'  # 输出文件名
data.to_csv(output_file, index=False)  # 保存为新的 CSV 文件，不包括索引

# 打印归一化后的数据（可选）
print(data[[column_to_normalize, column_to_normalize + '_normalized']])

# 输出成功信息
print(f"归一化结果已保存到文件: {output_file}")
