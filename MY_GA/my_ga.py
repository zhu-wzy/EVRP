# -*- coding: utf-8 -*-
import numpy as np
import math
import os
import heapq
from itertools import combinations
from matplotlib.collections import LineCollection
import pandas as pd
import random
import re
from tqdm import tqdm
import copy
import itertools
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
from collections import Counter
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch
import time


# 记录开始时间
start_time = time.time()
# 设置全局字体为支持中文的字体，例如 SimHei（黑体）
plt.rcParams['font.family'] = 'SimHei'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

#region 所有需求数据

# 计算距离矩阵
def calculate_distance_matrix(df_nodes):
    num_nodes = len(df_nodes)
    distance_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            # 计算欧几里得距离
            x1, y1 = df_nodes.loc[i, ['x', 'y']]
            x2, y2 = df_nodes.loc[j, ['x', 'y']]
            distance_matrix[i, j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return distance_matrix

# 将距离矩阵保存为 Excel 表格
def save_distance_matrix_to_excel(distance_matrix, df_nodes, output_file="distance_matrix.xlsx"):
    # 创建一个 DataFrame，用于存储距离矩阵
    # 行列索引使用节点的 id 或者 StringID
    ids = df_nodes['id'].tolist()  # 使用节点的 id 作为索引
    distance_df = pd.DataFrame(distance_matrix, index=ids, columns=ids)

    # 将 DataFrame 写入 Excel 文件
    distance_df.to_excel(output_file, index=True)
    print(f"距离矩阵已保存到 {output_file}")



# 读取数据
def read_evrptw_file(file_path):
    # 按行读取数据，按行保存在列表lines里
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 解析节点数据
    node_data = []
    node_section = True  # 用于判断是否在节点数据区域
    for line in lines:
        parts = line.strip().split() # 去除收尾空白字符，按空白分成单词
        # 将节点数据行放入node_data
        if len(parts) == 8 and re.match(r'^[A-Z]\d+', parts[0]):
            node_data.append(parts)
        elif len(parts) > 1 and '/' in parts[1]:
            node_section = False  # 遇到参数数据时，停止解析节点

    # 转换为DataFrame
    columns = ["StringID", "Type", "x", "y", "demand", "ReadyTime", "DueDate", "ServiceTime"]
    df_nodes = pd.DataFrame(node_data, columns=columns)
    # 转换数据类型
    numeric_cols = ["x", "y", "demand", "ReadyTime", "DueDate", "ServiceTime"]
    df_nodes[numeric_cols] = df_nodes[numeric_cols].astype(float) # 上述列中的数据转换为浮点类型

    # 客户点数目(含配送中心)
    dimension = df_nodes[df_nodes["Type"].isin(["c", "d"])].shape[0]# 获取行数筛选cd
    # 充电站数目
    num_of_stations = df_nodes[df_nodes["Type"].isin(["f"])].shape[0]# 获取行数筛选f
    # 修改 DueDate 值
    df_nodes.loc[df_nodes['Type'].isin(['d', 'f']), 'DueDate'] = 9999# 充电站、车站的时间不限

    # 调整顺序，将 Type = 'f' 的放到最后
    df_nodes = pd.concat([df_nodes[df_nodes['Type'] != 'f'], df_nodes[df_nodes['Type'] == 'f']]).reset_index(drop=True)

    df_nodes.insert(0, 'id', range(len(df_nodes)))# 插入id列0,1,2，。。。

    station_list = df_nodes.loc[df_nodes['Type'] == 'f', 'id'].tolist() # station列表储存车站id

    start_list = df_nodes.loc[df_nodes['Type'] == 'd', 'id'].tolist()
    # print(df_nodes)
    customer_list = df_nodes.loc[df_nodes['Type'] == 'c', 'id'].tolist()# 列表储存客户id
    # print(station_list)

    # 解析车辆参数
    battery_capacity = capacity = energy_consumption = speed = 0 # 初始化电池容量、载货量、能量消耗率、速度

    # 读取其他参数数据行
    for line in lines:
        match = re.match(r'([A-Za-z ]+) /([\d.]+)/', line.strip())#match
        if match:
            key, value = match.groups()
            value = float(value)
            if "fuel tank capacity" in key:
                battery_capacity = value
            elif "load capacity" in key:
                capacity = value
            elif "fuel consumption rate" in key:
                energy_consumption = value
            elif "average Velocity" in key:
                speed = value

    return df_nodes,  battery_capacity, capacity, energy_consumption, speed, dimension, num_of_stations, station_list, customer_list, start_list
           # 表格、     电池容量、          车容量、    能量消耗率、          速度、 车站、客户点数目、 充电站数目、   充电站列表、    车站、客户点列表

# file_path = "c101_21.txt"
file_path = "c101_21.txt"
#file_path = "c103C15"

(df_nodes, battery_capacity, capacity, energy_consumption, speed, dimension,
 num_of_stations, station_list, customer_list, start_list) = read_evrptw_file(file_path)
# print(df_nodes)

# 开始时间
eh = df_nodes['ReadyTime'].values.tolist()
# print(eh)
# 结束时间
lh = df_nodes['DueDate'].values.tolist()
# print(lh)
# 服务时间
h = df_nodes['ServiceTime'].values.tolist()
# 横坐标
X = df_nodes['x'].values.tolist()
# 纵坐标
Y = df_nodes['y'].values.tolist()
# df_nodes.to_excel("test.xlsx")
# print(df_nodes)
# print(battery_capacity, capacity, energy_consumption, speed, dimension, num_of_stations, station_list, customer_list)
distance_matrix = calculate_distance_matrix(df_nodes)
# print(distance_matrix)
# 调用函数保存距离矩阵
# save_distance_matrix_to_excel(distance_matrix, df_nodes, output_file="distance_matrix.xlsx")
# 创建一个需求的字典，id列为key，需求值为value
demands = df_nodes.set_index('id')['demand'].to_dict()
#print(demands)
# 所有点的数目
n_1 = dimension + num_of_stations
#endregion




# region 生成初始种群函数

def initialize_population(pop_size, customer_list, capacity, demands):
    population = []  # 总的种群

    for _ in range(pop_size):
        random.shuffle(customer_list)  # 随机打乱客户顺序
        individual = []  # 总的访问列表
        vehicle = []  # 车辆访问客户的列表

        for customer in customer_list:
            vehicle.append(customer)
            if sum(demands[i] for i in vehicle) > capacity:
                vehicle.pop(-1)  # 超过载重约束，回退上一个客户
                individual.append(vehicle)
                vehicle = [customer]  # 新建车辆路径
        if vehicle:
            individual.append(vehicle)  # 最后一辆车的路径

        population.append(individual)

    return population, individual


# 将初始种群进行编码（暂时没什么卵用）
# population = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# '1-2-3|4-5-6|7-8-9'
def convert_population_to_strings(population):
    return '|'.join('-'.join(str(customer) for customer in individual) for individual in population)
# endregion
"""
population, individual=initialize_population(2, customer_list, capacity, demands)
print(population)

print(individual)
"""



# region 计算成本函数
# route是变量individual列表中的元素
# individual=[[1, 4, 15, 8, 11, 6, 14, 12, 2, 10, 7], [13, 3, 9, 5]]
# 则 for route in individual 即可route=[1, 4, 15, 8, 11, 6, 14, 12, 2, 10, 7]
# 计算路径中的总需求
def calculate_route_revenue(route, demands):
    # 路径上的每个节点的需求都要加总，忽略起点和终点 (0)
    revenue = 0
    for node in route:
        if node != 0:  # 排除起点和终点 0 的需求
            revenue += demands.get(node, 0)  # 获取节点的需求，如果没有则默认为 0
    return revenue

# 计算剩余电量返回剩余电量、充电列表、距离列表
def calculate_remaining_battery(route, distance_matrix, energy_consumption, battery_capacity, station_list):
    # 初始化剩余电量为 battery_capacity（从0出发时，电量为满电）
    remaining_battery = [battery_capacity]  # 记录每个节点到达时的剩余电量
    current_battery = battery_capacity

    # 用于记录节点之间的距离和充电量列表
    distance_list = []
    charging_list = []

    # 遍历路径中的每个节点（从起点到终点）
    for i in range(1, len(route)):
        # 获取当前节点和下一个节点
        current_node = route[i - 1]
        next_node = route[i]

        # 计算当前节点到下一个节点的距离
        distance = distance_matrix[current_node][next_node]
        distance_list.append(distance)  # 记录节点之间的距离

        # 计算消耗的电量
        energy_used = distance * energy_consumption

        # 更新剩余电量
        current_battery -= energy_used

        # 如果到达充电站，电量恢复为 battery_capacity(完全充满电)
        if next_node in station_list:
            charging_amount = battery_capacity - current_battery  # 需要充电的量(满充策略)
            # current_battery = battery_capacity * 0.8
            current_battery = battery_capacity  # 恢复电池为满电（满充策略）
            # current_battery = battery_capacity * 0.8
        else:
            charging_amount = 0  # 如果不是充电站，则充电量为0

        # 确保电量不会低于 0
        current_battery = max(current_battery, 0)

        # 记录当前节点到达时的剩余电量
        remaining_battery.append(current_battery)

        # 将充电量记录到充电量列表
        charging_list.append(charging_amount)

    return remaining_battery, distance_list, charging_list

# 计算载重减少返回载重减少列表（当前载重的列表）
def calculate_weight_reduction(route, demands):
    weight_reduction_list = []

    # 初始化：根据route路径中的客户，计算初始载重
    # print(route)
    total_demand = sum(demands[node] for node in route[1:-1])  # 初始载重为除了起点和终点外的所有节点的需求和
    weight_reduction_list.append(total_demand)  # 初始载重

    # 计算每次移动时的载重变化
    for i in range(1, len(route) - 1):  # 从1到len(route)-1，遍历所有节点
        total_demand -= demands[route[i]]  # 当前节点的需求从载重中减去

        # 记录当前路径的载重
        weight_reduction_list.append(total_demand)

    return weight_reduction_list


# 计算时间窗惩罚成本
def calculate_penalty_cost(arrival_time, eh_date, lh_date, h_date, epu, lpu):
    """计算时间窗惩罚成本，返回离开时间和惩罚成本"""
    if arrival_time < eh_date:  # 早到，需要等待且有惩罚
        departure_time = eh_date + h_date  # 不能提前服务
        penalty_cost = (eh_date - arrival_time) * epu
    elif arrival_time > lh_date:  # 晚到，有惩罚
        # print(arrival_time, lh_date, lpu)
        penalty_cost = lpu * (arrival_time - lh_date)  # 计算晚到惩罚
        departure_time = arrival_time + h_date  # 晚到但可服务
    else:  # 按时到达
        departure_time = arrival_time + h_date  # 正常服务
        penalty_cost = 0  # 无惩罚
    return departure_time, penalty_cost

# 计算路程行驶成本
def calculate_distance(distance, P4, weight_reduction_list_per):
    return distance * P4 * weight_reduction_list_per

# 计算路程行驶时间 输出时间行驶total_time单位是秒
def calculate_travel_time(distance, start_time, speed_data_file='speed_normalized.csv'):
    # 读取数据
    data = pd.read_csv(speed_data_file)

    # 提取时间和速度数据
    time_ids = data['time_id'].values  # 时间 ID
    speeds = data['speed'].values        # 速度

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
#distance = distance_matrix[0][1]
#print(distance)
#time = calculate_travel_time(distance, start_time=200, speed_data_file='speed_normalized.csv')
#print(time)

# 计算时间窗惩罚成本和路程成本
def vrptw_simulation(path, distance_matrix, eh, lh, h, P4,  weight_reduction_list, epu, lpu):
    """模拟 VRPTW 路径计算，返回、惩罚成本、速度、到达时间、离开时间的列表"""

    path_distance_costs = []
    path_penalty_costs = []
    path_speeds = []
    path_arrival_times = []
    path_departure_times = []

    # 初始节点（配送中心）
    current_time = 0
    total_cost = 0
    arrival_time_list = []
    departure_time_list = []
    penalty_cost_list = []
    distance_penalty_cost_list = []

    for i in range(len(path) - 1):
        cur_node = path[i]
        #print(i)
        next_node = path[i + 1]
        distance = distance_matrix[cur_node][next_node]  # 读取距离
        weight_reduction_list_per = weight_reduction_list[i]

        P_cost = P4[2]
        #print(current_time)
        current_time = math.ceil(current_time)
        #print(current_time)
        travel_time = calculate_travel_time(distance, start_time=current_time, speed_data_file='speed_normalized.csv')
        #print(travel_time)
        travel_time = round(travel_time)
        #print(travel_time)
        arrival_time = current_time + travel_time
        #print(arrival_time)
        departure_time, penalty_cost = calculate_penalty_cost(arrival_time, eh[next_node], lh[next_node], h[next_node], epu, lpu)
        distance_cost = calculate_distance(distance, P_cost, weight_reduction_list_per)
        distance_penalty_cost = distance_cost + penalty_cost
        #print(distance_penalty_cost)
        total_cost = distance_penalty_cost + total_cost
        #print(total_cost)
        # 更新当前时间
        current_time = arrival_time

        # 记录每个站点的数据
        arrival_time_list.append(arrival_time)
        departure_time_list.append(departure_time)
        penalty_cost_list.append(penalty_cost)
        distance_penalty_cost_list.append(distance_penalty_cost)

    return arrival_time_list, departure_time_list, penalty_cost_list, distance_penalty_cost_list, total_cost
"""测试代码
P4 = {1: 0.05, 2: 0.1}
    epu, lpu = 1, 1  # 时间窗成本
    route = individual[0]
    print(route)
    weight_reduction_list = calculate_weight_reduction(route, demands)
    arrival_time_list, departure_time_list, penalty_cost_list, distance_penalty_cost_list, total_cost = \
        vrptw_simulation(route, distance_matrix, eh, lh, h, P4,  weight_reduction_list, epu, lpu)
    print(arrival_time_list)
    print(departure_time_list)
    print(total_cost)
    print("finish")
"""

# 计算一个个体上的总成本(为插入充电站节点因此输出的充电量列表为0)
def fitness_evaluation(individual, demands, distance_matrix, energy_consumption, battery_capacity,
                                                        station_list, eh, lh, h):

    P1 = 50  # 单价系数：货物的单位价格
    P2 = 100  # 车辆系数
    P3 = 1  # 充电成本

    P4 = {1: 0.05, 2: 0.1}  # 不同道路成本单价
    epu, lpu = 1, 1  # 时间窗成本

    # 计算每条路径的收益与成本
    sum_revenue = 0

    for route in individual:
        _route = [0] + route + [0]
        # print(_route)

        # 该路径的收益
        revenue = calculate_route_revenue(_route, demands) * P1
        # print("收益:", revenue)

        # 车辆成本
        per_vehicle_cost = P2
        # print("车辆成本：", per_vehicle_cost)

        # 计算剩余电量、距离列表和充电量列表
        remaining_battery, distance_list, charging_list = calculate_remaining_battery(_route, distance_matrix,
                                                                                      energy_consumption,
                                                                                      battery_capacity, station_list)

        # 打印每个节点到达时的剩余电量、距离和充电量
        # print("每个节点到达时的剩余电量：", remaining_battery)
        # print("节点之间的距离：", distance_list)
        # print("充电量列表：", charging_list)

        charging_list_cost = sum(charging_list) * P3
        # print("充电成本：", charging_list_cost)

        # 计算载重减少的列表  --- 注意修改
        weight_reduction_list = calculate_weight_reduction(_route, demands)

        # 打印载重减少的列表, 对应路径中的变化，用于计算配送路径成本和惩罚成本
        # print("载重减少列表：", weight_reduction_list)

        # 用来存储每条路径的最优结果
        all_distance_costs, all_penalty_costs, all_speeds, all_arrival_times, all_departure_times = vrptw_simulation(
            _route, distance_matrix, eh, lh, h, P4,  weight_reduction_list, epu, lpu
        )

        # **输出结果**
        # print("\n距离成本列表：")
        # print(all_distance_costs)
        # print("\n惩罚成本列表：")
        # print(all_penalty_costs)
        # print("\n最优速度列表：")
        # print(all_speeds)
        # print("\n到达时间列表：")
        # print(all_arrival_times)
        # print("\n离开时间列表：")
        # print(all_departure_times)

        # 计算总收益
        sum_revenue = sum_revenue + per_vehicle_cost + charging_list_cost + sum(all_penalty_costs) + sum(all_distance_costs)
        #print(f"车辆成本+充电成本+时间惩罚成本+路程成本：{sum_revenue}")
    return sum_revenue
"""测试代码
P4 = {1: 0.05, 2: 0.1}
    epu, lpu = 1, 1  # 时间窗成本
    route = individual[0]
    print(individual)
    weight_reduction_list = calculate_weight_reduction(route, demands)
    sum_revenue = fitness_evaluation(individual, demands, distance_matrix, energy_consumption, battery_capacity,
                                                        station_list, eh, lh, h)
    print(sum_revenue)
"""



# endregion



# region路径处理函数(插入充电站节点、并计算成本)
# 计算
def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    extended_route = route
    if route[0] != 0 or route[-1] != 0:
        extended_route = [0] + route + [0]
    for i in range(len(extended_route) - 1):
        total_distance += distance_matrix[extended_route[i]][extended_route[i + 1]]
    return total_distance


# 找到最近的充电站
def find_nearest_station(node_no, distance_matrix, station_list):
    return int(min(station_list, key=lambda station_no: distance_matrix[int(node_no)][int(station_no)]))

# 找到一条路径内路径最短的情况
def two_opt(route, distance_matrix):
    _route = [0] + route + [0]

    improved = True
    while improved:
        improved = False
        #print(f"输出变量：{len(_route) - 2}")
        for i in range(1, len(_route) - 2):
            for j in range(i + 1, len(_route) - 1):
                old_cost = distance_matrix[_route[i - 1]][_route[i]] + distance_matrix[_route[j]][_route[j + 1]]
                new_cost = distance_matrix[_route[i - 1]][_route[j]] + distance_matrix[_route[i]][_route[j + 1]]
                if new_cost < old_cost:
                    _route[i:j + 1] = reversed(_route[i:j + 1])
                    improved = True
    _route = _route[1:-1]
    return _route
"""测试代码
for route in individual:
        print(f"优化前：{route}")
        _route = two_opt(route, distance_matrix)
        print(f"优化后：{_route}")
"""


# 路径内修复，插入充电站
def simple_repair(initial_route, battery_capacity, energy_consumption, distance_matrix, station_list):
    route = initial_route.copy()

    if len(route) == 0 or len(route) == 1:
        return route
    repaired_route = [route.pop(0)]# ‘取出’第一个元素
    current_node = repaired_route[-1]# 得到最后一个元素
    next_node = route[0]# 得到第一个元素

    energy_left = battery_capacity - energy_consumption * distance_matrix[0][current_node]# 0.8

    while route:
        updated_energy_left = energy_left - energy_consumption * distance_matrix[current_node][next_node]
        nearest_station_to_next = find_nearest_station(next_node, distance_matrix, station_list)
        # 电量大于下个节点到充电站的耗电量，将下个节点收入修补路径列表
        if updated_energy_left >= energy_consumption * distance_matrix[next_node][nearest_station_to_next]:
            repaired_route.append(next_node)
            route.pop(0)
            next_node = route[0] if len(route) > 0 else 0
            current_node = repaired_route[-1]
            energy_left = updated_energy_left
        else:# 电量小于下个节点到充电站的耗电量
            # 寻找当前节点可以到达的充电站
            reachable_stations_from_current = [int(station) for station in station_list if
                                               energy_left >= energy_consumption * distance_matrix[current_node][
                                                   int(station)]]
            # 如果当前节点没有可以到达的充电站
            if not reachable_stations_from_current:
                index = repaired_route.index(current_node)
                prev_node = 0 if index == 0 else repaired_route[index - 1]
                # 找到上一个节点，计算上一个节点的能量消耗
                prev_energy_left = energy_left + energy_consumption * distance_matrix[current_node][prev_node]
                # 找上一个节点可以到达的充电站
                reachable_stations_from_prev_node = [int(station) for station in station_list if
                                                     prev_energy_left >= energy_consumption *
                                                     distance_matrix[prev_node][int(station)]]
                # 找最近的充电站
                _nearest_station_to_current_node = find_nearest_station(current_node, distance_matrix,
                                                                        reachable_stations_from_prev_node)
                # 将最近的充电站插入原来节点的位置
                repaired_route.insert(index, _nearest_station_to_current_node)
                # energy_left = 0.8*battery_capacity - energy_consumption * distance_matrix[_nearest_station_to_current_node][
                #     current_node]
                energy_left = battery_capacity - energy_consumption * distance_matrix[_nearest_station_to_current_node][
                     current_node]

                reachable_stations_from_current = [int(station) for station in station_list if
                                                   energy_left >= energy_consumption * distance_matrix[current_node][
                                                       int(station)]]
            # 如果当前节点有可以到达的充电站则找到最近的充电站
            nearest_station_to_next = find_nearest_station(next_node, distance_matrix, reachable_stations_from_current)
            # 将当前节点的充电站插入列表
            repaired_route.append(nearest_station_to_next)
            # energy_left = battery_capacity*0.8
            energy_left = battery_capacity  # 充电策略选择0.8
            current_node = repaired_route[-1]

    # 如果剩余电量小于当前节点到达车站的耗电量
    if energy_left < energy_consumption * distance_matrix[current_node][0]:
        # 找到当前节点的最近车站
        nearest_station_to_current = find_nearest_station(current_node, distance_matrix, station_list)
        repaired_route.append(nearest_station_to_current)

    # 返回修复后的路径
    return repaired_route
""" 测试代码
for route in individual:
        print(route)
        repaired_route = simple_repair(route, battery_capacity, energy_consumption, distance_matrix, station_list)
        print(repaired_route)
"""


# 计算引入充电站后的成本计算和individual修复
def local_search(individual, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h):
    battery_capacity = copy.deepcopy(battery_capacity)  # 深拷贝不会修改原来的值
    energy_consumption = energy_consumption
    distance_matrix = distance_matrix
    station_list = station_list

    # print("初始解", individual)
    # 2-opt
    optimized_individual = []
    for route in individual:
        if len(route) <= 1:
            optimized_individual.append(route)
            continue
        route_01 = two_opt(route, distance_matrix)

        shuffled_route = route[:]  # 浅拷贝含有可变对象（列表、字典、集合、数组）时与深拷贝不同
        random.shuffle(shuffled_route)

        route_02 = two_opt(shuffled_route, distance_matrix)

        route_candidates = [route_01, route_02]

        best_route = min(route_candidates, key=lambda x: calculate_total_distance(x, distance_matrix))
        optimized_individual.append(best_route)

    original_individual = optimized_individual
    # print("2opt优化解", original_individual)

    # 插入充电站之后的individual
    repaired_individual = []
    for route in original_individual:
        repaired_route = simple_repair(route, battery_capacity, energy_consumption, distance_matrix, station_list)
        repaired_individual.append(repaired_route)

    # repaired_individual = [[2,5,1,4,3]]
    # print("计算成本的解", repaired_individual)

    # 计算插入充电站之后的总成本
    cost = fitness_evaluation(repaired_individual, demands, distance_matrix, energy_consumption, battery_capacity, station_list, eh, lh, h)

    return (repaired_individual, cost)


# endregion（（）（）


"""
以下模块为模型修改2025/4/11
遗传算法模块：
1、适应度计算函数
2、选择算子
3、交叉变换算子
4、遗传算法主算法以及可视化部分
"""

# region 遗传算法部分(修改模块)
# 适应度计算函数
# 约束条件：
# 1.同一条路径不能有相同的节点，若有相同节点则惩罚值增加inf
# 2.每个个体必须访问所有节点
# 3.载重约束

def constraint_punishment1(individual):

    for route in individual:
        counter = Counter(route)
        # 相同节点约束
        if any(count > 1 for count in counter.values()):
            punishment = float("inf")
            #print("存在重复节点")
        else:
            punishment = 0
            #print("没有重复节点")
    return punishment


def constraint_punishment2(individual):
    customer_list_real = []
    for route in individual:
        for i in route:
            customer_list_real.append(i)
    customer_list_real.sort()
    customer_list.sort()
    if customer_list_real == customer_list:
        punishment2 = 0
        #print("全部客户点均已访问")
    else:
        punishment2 = float("inf")
        #print("存在未访问客户")
    return punishment2

def constraint_punishment3(individual, capacity=capacity,demands=demands):
    for route in individual:
        demands_sum = 0
        for i in route:
            demands_sum = demands_sum + demands[i]
        if demands_sum > capacity:
            punishment3 = float("inf")
            #print("超出了载重约束")
            break
        else:
            punishment3 = 0
            #print("没有超出载重约束")
    return punishment3


# 计算适应度=成本（cost）+约束惩罚（punishment）
def fitness_calculation(individual, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h):
    (repaired_individual, cost) = local_search(individual, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h)
    punishment = constraint_punishment1(repaired_individual)
    punishment2 = constraint_punishment2(individual)
    punishment3 = constraint_punishment3(individual, capacity=capacity,demands=demands)
    fitness = cost + punishment + punishment2 + punishment3

    return fitness


# 选择算子1（精英保留策略）
def selection_find_best_individual(population, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h):
    fitness_best = float("inf")

    for individual in population:
        fitness = fitness_calculation(individual, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h)
        if fitness < fitness_best:
            individual_best = individual
            fitness_best = fitness
        else:
            pass
    #print(f"精英个体：{individual_best},精英个体适应度：{fitness_best}")
    return individual_best, fitness_best

# 选择算子2（轮盘赌选择） 选择个体和初始种群数量相同
def selection_roulette_wheel(population, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h):
    fitness_list = []
    for individual in population:
        fitness = fitness_calculation(individual, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h)
        fitness_list.append(fitness)

    # 取适应度的倒数，避免除零错误
    reciprocal_fitness_list = []
    for fitness in fitness_list:
        if fitness == 0:
            # 当适应度为 0 时，设置一个较大的值，避免除零错误
            reciprocal_fitness_list.append(float('inf'))
        else:
            reciprocal_fitness_list.append(1 / fitness)



    # 计算倒数适应度总和
    total_reciprocal_fitness = sum(reciprocal_fitness_list)
    if total_reciprocal_fitness == float('inf'):
        raise ValueError("所有个体的适应度值都为 0，无法进行选择。")

    # 计算每个个体的选择概率
    probabilities = [reciprocal_fitness / total_reciprocal_fitness for reciprocal_fitness in reciprocal_fitness_list]

    # 计算累积概率
    cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]


    # 选择个体
    selected_individuals = []
    for _ in range(len(population)-1):# 引入精英保留种群
        random_num = random.random()
        for i, cumulative_prob in enumerate(cumulative_probabilities):
            if random_num <= cumulative_prob:
                selected_individuals.append(population[i])
                break
    individual_best, _ = selection_find_best_individual(population, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h)
    selected_individuals.append(individual_best)
    return selected_individuals




# 变异算子
# 变异算子会出现访问相同客户点的情况降低个体适应度因此只考虑使用交叉算子
# 染色体匹配算子
def random_pairing(population):
    # 复制种群列表，避免修改原始列表
    population_copy = population.copy()
    # 打乱种群列表的顺序
    random.shuffle(population_copy)
    pairs = []
    # 遍历打乱后的种群列表，两两分组
    for i in range(0, len(population_copy), 2):
        if i + 1 < len(population_copy):
            pairs.append([population_copy[i], population_copy[i + 1]])
    return pairs

# 变异算子
def random_swap(lst):
    # 确定交换次数，范围是 1 到列表长度的一半
    swap_count = random.randint(1, len(lst) // 2)
    for _ in range(swap_count):
        # 随机选择两个不同的位置
        index1, index2 = random.sample(range(len(lst)), 2)
        # 交换这两个位置的元素
        lst[index1], lst[index2] = lst[index2], lst[index1]
    return lst



# 染色体顺序交叉变异算子
def order_crossover(parent1, parent2):
    size = len(parent1)
    if len(parent2) != size:
        print(f"parent1{parent1}")
        print(f"parent2{parent2}")
        raise ValueError("Parents must be of the same length")

    # 随机选择交叉的起始和结束位置
    start = random.randint(0, size - 1)
    end = random.randint(start, size - 1)

    # 初始化子代
    child1 = [None] * size
    child2 = [None] * size

    # 将父代的中间段复制到子代
    child1[start:end+1] = parent1[start:end+1]
    child2[start:end+1] = parent2[start:end+1]

    # 填充子代1的剩余位置
    ptr = (end + 1) % size
    for i in list(range(end + 1, size)) + list(range(0, start)):
        while parent2[ptr] in child1:
            ptr = (ptr + 1) % size
        child1[i] = parent2[ptr]
        ptr = (ptr + 1) % size

    # 填充子代2的剩余位置
    ptr = (end + 1) % size
    for i in list(range(end + 1, size)) + list(range(0, start)):
        while parent1[ptr] in child2:
            ptr = (ptr + 1) % size
        child2[i] = parent1[ptr]
        ptr = (ptr + 1) % size

    if child1 == parent1:
        child1 = random_swap(child1)

    if child2 == parent2:
        child2 = random_swap(child2)



    return child1, child2


# 分割生成子代个体
def child_stroke(child, capacity, demands):
    individual_child = []
    vehicle = []

    for customer in child:
        vehicle.append(customer)

        if sum(demands[i] for i in vehicle) > capacity:
            vehicle.pop(-1)  # 超过载重约束，回退上一个客户
            individual_child.append(vehicle)
            vehicle = [customer]  # 新建车辆路径
    if vehicle:
            individual_child.append(vehicle)  # 最后一辆车的路径

    return individual_child


# 选择后代生成新种群
def new_population(population, capacity, demands):
    capacity = capacity
    demands = demands
    pairs = random_pairing(population)

    #print(pairs)

    population_child = []
    for pair in pairs:
        individual_parent1 = pair[0]
        individual_parent2 = pair[1]
        parent1 =list(itertools.chain(*individual_parent1))
        parent2 =list(itertools.chain(*individual_parent2))
        #print(f"parent1={parent1},parent2={parent2}")
        child1, child2 = order_crossover(parent1, parent2)
        #print(f"child1={child1},child2={child2}")
        individual_child1 = child_stroke(child1, capacity, demands)
        individual_child2 = child_stroke(child2,capacity,demands)
        #print(f"individual_child1={individual_child1},individual_child2={individual_child2}")
        population_child.append(individual_child1)
        population_child.append(individual_child2)


    return population_child

# 定义 VNS 处理函数
# VNS处理函数：按规则交换个体内所有路径的所有点的位置
def vns_process(population):
    new_population = []
    for individual in population:
        new_individual = []
        for route in individual:
            new_route = route[::-1]  # 按规则交换，这里采用反转列表的方式
            new_individual.append(new_route)
        new_population.append(new_individual)
    return new_population



# 遗传算法主函数(包含可视化部分)
def GA_main(pop_size, generations):
    # 初始化初始种群
    population, _ = initialize_population(pop_size, customer_list, capacity, demands)
    print(population)
    fitness_best_list=[]
    fitness_best = 0
    individual_best = []
    individual_best_of_all = []
    fitness_best_of_all = float("inf")
    stagnant_generations = 0  # 记录最佳适应度未改变的代数


    # 主循环
    for i in range(generations):
        population_parent = []

        # 找到种群中最佳个体，并用轮盘赌选出其余个体作为父代
        individual_best, fitness_best = selection_find_best_individual(population, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h)

        # 只有当寻找到的适应度小于最优适应度时，才会改变
        if fitness_best < fitness_best_of_all:
            fitness_best_of_all = fitness_best
            fitness_best_list.append(fitness_best_of_all)
            individual_best_of_all = individual_best
            stagnant_generations = 0  # 重置停滞代数
        else:
            fitness_best_list.append(fitness_best_of_all)
            stagnant_generations += 1  # 停滞代数加 1


        population_parent.append(individual_best)
        #print(population_parent)
        selected_individuals = selection_roulette_wheel(population, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h)
        population_parent = selected_individuals

       #print(f"11111111111111111{population_parent}")


        # 用父代生成子代
        population_child = new_population(population_parent, capacity, demands)
         # 检查 population_child 是否发生变化
        if len(population_child) != pop_size:
            raise ValueError(f"population_child 发生了变化，当前长度为: {len(population_child)},"
                             f"父代种群：{population_parent}"
                             f"迭代数{i}")

        if len(population_child) == 0:
            print("Warning: population_child is empty!")
            continue
        #print(population_child)
        #print(len(population_child))
        # 子代保存父代中的优秀个体
        # 生成一个随机索引

        # 当最佳适应度超过 10 代不发生改变时，进行 VNS 处理
        if stagnant_generations >= 10:
            population_child = vns_process(population_child)
            stagnant_generations = 0  # 重置停滞代数

        population_child[0] = individual_best
        population = population_child
        individual_best, fitness_best = selection_find_best_individual(population_child, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h)

        if i % 10 == 0:
            print(f"当前迭代{i}")

    repaired_individual_best, cost = local_search(individual_best_of_all, battery_capacity, energy_consumption, distance_matrix, station_list, demands, eh, lh, h)


    # 可视化部分代码：
    # 适应度变化曲线：
    y_data = fitness_best_list
    x_data = list(range(len(y_data)))
    # 绘制函数图，设置线条颜色为红色，样式为虚线，标记为圆形
    plt.plot(x_data, y_data, 'r--o')
    # 设置图形属性
    plt.title('适应度变化曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值（成本）')
    # 添加网格线
    plt.grid(True)
    # 显示图形
    plt.show()

    return individual_best_of_all, fitness_best, fitness_best_list, repaired_individual_best, cost, population

# endregion




# 测试主函数
if __name__ == '__main__':
    pop_size = 30
    generations = 150
    population, _ = initialize_population(pop_size, customer_list, capacity, demands)


    print("start")

    individual_best, fitness_best, fitness_best_list, repaired_individual_best, cost, population = GA_main(pop_size, generations)
    print(fitness_best_list)
    print(repaired_individual_best)
    print(len(population))
     # 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))
    # 绘制节点
    for i in start_list:
        plt.scatter(X[i], Y[i], color='red', s=50)
        depot_XY = (X[i], Y[i])

    for i in station_list:
        if (X[i], Y[i]) != depot_XY:
            plt.scatter(X[i], Y[i], color='blue', s=50, marker='^')

    for i in customer_list:
        plt.scatter(X[i], Y[i], color='green', s=50, marker='s')
    # 绘制路线：
    cmap = plt.get_cmap('nipy_spectral')
    num_routes = len(repaired_individual_best)
    colors = [cmap(i / num_routes) for i in range(num_routes)]
    n = 0
    for route in repaired_individual_best:
        _route = [0] + route + [0]
        x_coords = [X[i] for i in _route]
        y_coords = [Y[i] for i in _route]
        color = colors[n]
        n = n + 1
        for j in range(len(x_coords) - 1):
            start_x, start_y = x_coords[j], y_coords[j]
            end_x, end_y = x_coords[j + 1], y_coords[j + 1]
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            dx = end_x - start_x
            dy = end_y - start_y
            arrow = FancyArrowPatch((start_x, start_y), (mid_x, mid_y),
                                    arrowstyle='->', color=color, mutation_scale=20)
            ax.add_patch(arrow)
            plt.plot([start_x, end_x], [start_y, end_y], color=color, linestyle='-')



    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Depot', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='^', color='w', label='Charge_station', markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='s', color='w', label='Customer', markerfacecolor='green', markersize=10)
    ]
     # 设置图形属性
    plt.title('电动车路径规划可视化')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.legend()

    # 显示图形
    plt.show()

    print(f"最佳行驶方案{repaired_individual_best}")
    print(f"最佳车辆数{len(repaired_individual_best)}")
    print(f"最低成本{cost}")

    # 记录结束时间并计算运行时间
    end_time = time.time()
    run_time = end_time - start_time
    print(f"代码运行时间: {run_time} 秒")

    print("finish")


