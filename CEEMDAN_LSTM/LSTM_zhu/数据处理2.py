import pandas as pd

# 指定输入 CSV 文件的路径
input_file = 'CEEMDAN.csv'  # 替换为你的输入 CSV 文件路径

# 读取 CSV 文件
df = pd.read_csv(input_file)

# 检查列的数量（确保为 10 列）
if df.shape[1] != 11:
    print("CSV 文件中不包含 10 列数据。请检查文件。")
else:
    # 遍历每一列并保存为单独的 CSV 文件
    for i in range(11):
        # 取出每一列的数据并创建新的 DataFrame
        col_data = df.iloc[:, [i]]

        # 生成输出文件名
        output_file = f'IMF_{i + 0}.csv'  # 保存为 column_1.csv, column_2.csv, ..., column_10.csv

        # 保存数据到 CSV 文件，index=False 表示不保存行索引
        col_data.to_csv(output_file, index=False)
        print(f'Saved {output_file}')
