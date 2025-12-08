import numpy as np
import my_ga
import plotly.graph_objects as go

# 创建一个距离矩阵（示例数据）
distance_matrix = my_ga.distance_matrix

# 将矩阵中的值转换为字符串，用于显示在热力图上
text_matrix = np.round(distance_matrix, 2).astype(str)  # 保留两位小数

# 绘制热力图
fig = go.Figure(data=go.Heatmap(
    z=distance_matrix,
    colorscale='Viridis',
    text=text_matrix,  # 在每个点处显示数据值
    hoverinfo="text"   # 设置鼠标悬停时显示数据值
))

fig.update_layout(
    title="Distance Matrix Heatmap",
    xaxis_title="Point Index",
    yaxis_title="Point Index"
)

# 保存为静态图（PNG 格式）
fig.write_image("distance_matrix_heatmap.png")  # 保存为 PNG 文件
