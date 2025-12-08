import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_travel_time(distance, start_time=200, speed_data_file='speed_normalized.csv'):
    # 读取数据
    data = pd.read_csv(speed_data_file)

    # 提取时间和速度数据
    time_ids = data['time_id'].values  # 时间 ID
    speeds = data['speed_normalized'].values  # 速度

    # 将时间 ID 转换为分钟（假设 time_id 是从 1 开始的，每个 ID 代表 5 分钟）
    time_minutes = time_ids * 5

    # 计算行驶时间的逻辑
    remaining_distance = distance
    total_time = 0

    for i in range(len(time_minutes) - 1):
        speed = speeds[i]
        if start_time < time_minutes[i + 1]:
            effective_time = time_minutes[i + 1] - start_time
            distance_in_segment = speed * (effective_time * 60)

            if remaining_distance > distance_in_segment:
                total_time += effective_time * 60
                remaining_distance -= distance_in_segment
                start_time = time_minutes[i + 1]
            else:
                total_time += (remaining_distance / speed) * 60
                break
        else:
            continue

    return total_time

if __name__ == '__main__':
    # 示例：计算特定距离的行驶时间
    distance_to_travel = 1000  # 假设要行驶的路程为1000米
    travel_time = calculate_travel_time(distance_to_travel)

    # 输出行驶时间
    print(f"Total travel time for {distance_to_travel} meters: {travel_time:.2f} seconds")

    # 可视化分段速度函数
    data = pd.read_csv('speed_normalized.csv')
    time_ids = data['time_id'].values
    speeds = data['speed_normalized'].values
    time_minutes = time_ids * 5

    plt.figure(figsize=(12, 6))
    plt.grid(axis='y', linestyle='--', linewidth=0.7)

    plt.step(time_minutes, speeds, where='post', color='blue', linewidth=2, label='Speed (km/h)')

    end_time = 200 + (travel_time / 60)  # 转换为分钟
    plt.axvline(x=200, color='red', linestyle='--', linewidth=2, label='Start Time')
    plt.axvline(x=end_time, color='red', linestyle='--', linewidth=2, label='End Time')

    plt.xlabel('Time (HH:MM)')
    plt.ylabel('Speed (km/h)')

    # 设置 x 轴刻度
    x_ticks = np.arange(0, int(max(time_minutes) + 1), 60)
    x_tick_labels = [f"{(8 + (i // 60)) % 24}:00" for i in x_ticks]
    plt.xticks(x_ticks, x_tick_labels)

    # 设置 y 轴刻度
    plt.ylim(0, 2)  # 设置 y 轴范围
    plt.yticks(np.arange(0, 2.1, 0.1))  # 设置 y 轴刻度间隔为 0.1

    plt.legend()

    plt.show()
