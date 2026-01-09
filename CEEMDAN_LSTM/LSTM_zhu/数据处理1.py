import pandas as pd
import matplotlib.pyplot as plt

# 指定输入 XLSX 文件和输出 CSV 文件的路径
input_file = '实际结果.xlsx'  # 替换成你的 XLSX 文件路径
output_file = '实际结果.csv'  # 输出CSV文件路径

# 读取 XLSX 文件
df = pd.read_excel(input_file)

# 保存为 CSV 文件
df.to_csv(output_file, index=False, encoding='utf-8')  # 使用 utf-8 编码可以避免中文乱码









