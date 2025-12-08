"""

def calculate_travel_time(current_node, next_node, distance, speed_function):

    计算从当前节点到下一个节点的行驶时间。

    :param current_node: 当前节点编码（整数）
    :param next_node: 下一个节点编码（整数）
    :param distance: 两地之间的距离（浮点数）
    :param speed_function: 两地之间的速度分段函数，格式为 [(t1, v1), (t2, v2), ...]
                           其中 t1, t2 是时间段的长度，v1, v2 是对应时间段内的速度。
    :return: 行驶时间（浮点数）

    remaining_distance = distance  # 剩余距离
    total_time = 0  # 总行驶时间

    # 遍历速度分段函数
    for segment in speed_function:
        time_interval, speed = segment  # 当前时间段的长度和速度

        # 计算当前时间段内能行驶的距离
        distance_in_segment = speed * time_interval

        if remaining_distance <= distance_in_segment:
            # 如果剩余距离可以在当前时间段内完成
            total_time += remaining_distance / speed
            return total_time
        else:
            # 如果当前时间段无法完成剩余距离
            total_time += time_interval
            remaining_distance -= distance_in_segment

    # 如果速度函数的时间段用完了，但距离还没走完
    if remaining_distance > 0:
        raise ValueError("速度分段函数不足以覆盖整个行程距离！")

    return total_time
"""


"""
print(f"客户点列表{my_ga.customer_list}")

population, individuals = my_ga.initialize_population(2, my_ga.customer_list, my_ga.capacity, my_ga.demands)
customer_list_real = []
for individual in population:
    customer_list_real = []
    print(f"个体{individual}")
    for route in individual:
        for i in route:
            customer_list_real.append(i)
    customer_list_real.sort()
    my_ga.customer_list.sort()
    print(f"实际访问列表{customer_list_real}")
    print(f"客户点列表{my_ga.customer_list}")
    if customer_list_real == my_ga.customer_list:
        punishment = 0
        print("全部客户点均已访问")
    else:
        punishment = float("inf")

print(punishment)
"""
"""
a = 1
def sum(b,a=a):
    c = a +b
    return c


if __name__ == '__main__':
    c = sum(1,a=a)
    print(c)
"""


import random
import my_ga
import itertools





def random_pairing(population):

    # 复制种群列表，避免修改原始列表
    population_copy = population.copy()
    # 打乱种群列表的顺序
    random.shuffle(population_copy)
    pairs = []
    # 遍历打乱后的种群列表，两两分组
    for i in range(0, len(population_copy), 2):
        if i + 1 < len(population_copy):
            pairs.append((population_copy[i], population_copy[i + 1]))
    return pairs


def order_crossover(parent1, parent2):

    # 随机选择两个交叉点
    start, end = sorted(random.sample(range(len(parent1)), 2))
    # 初始化子代个体
    child1 = [-1] * len(parent1)
    child2 = [-1] * len(parent2)
    # 将父代 1 交叉段复制到子代 1
    child1[start:end] = parent1[start:end]
    # 将父代 2 交叉段复制到子代 2
    child2[start:end] = parent2[start:end]
    # 填充子代 1 剩余部分
    index = 0
    for i in range(len(parent2)):
        if parent2[i] not in child1[start:end]:
            while child1[index] != -1:
                index = (index + 1) % len(child1)
            child1[index] = parent2[i]
            index = (index + 1) % len(child1)
    # 填充子代 2 剩余部分
    index = 0
    for i in range(len(parent1)):
        if parent1[i] not in child2[start:end]:
            while child2[index] != -1:
                index = (index + 1) % len(child2)
            child2[index] = parent1[i]
            index = (index + 1) % len(child2)
    return child1, child2

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








if __name__ == '__main__':
    # capacity = my_ga.capacity
    # demands = my_ga.demands
    # child = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # mid = child_stroke(child, capacity, demands)
    # print(mid)

   # capacity = my_ga.capacity
   # demands = my_ga.demands
   # population, individual = my_ga.initialize_population(4, my_ga.customer_list, my_ga.capacity, my_ga.demands)
   # pairs = random_pairing(population)
   # print(pairs)
   # population_child = []
   # for pair in pairs:
   #     individual_parent1 = pair[0]
   #     individual_parent2 = pair[1]
   #     parent1 =list(itertools.chain(*individual_parent1))
   #     parent2 =list(itertools.chain(*individual_parent2))
   #     print(f"parent1={parent1},parent2={parent2}")
   #     child1, child2 = order_crossover(parent1, parent2)
   #     print(f"child1={child1},child2={child2}")
   #     individual_child1 = child_stroke(child1, capacity, demands)
   #     individual_child2 = child_stroke(child2,capacity,demands)
   #     print(f"individual_child1={individual_child1},individual_child2={individual_child2}")
    parent1 = [1,2,3,4,5,6,7]
    parent2 = [1,2,3,4,5,6]
    child1, child2 = order_crossover(parent1, parent2)
    print(child1)
    print(child2)






"""
顺序交叉（Order Crossover, OX）是遗传算法中用于解决组合优化问题（如旅行商问题 TSP、车辆路径问题 VRP）的一种经典交叉算子。其核心目标是在两个父代个体的基础上，生成两个子代个体，同时保证子代个体中元素的唯一性和完整性。下面详细解释其原理。
原理步骤
1. 随机选择交叉点
从父代个体的基因序列中随机选择两个不同的位置作为交叉点，这两个交叉点将父代个体的基因序列划分为三个部分：交叉点之前的部分、交叉点之间的部分和交叉点之后的部分。
2. 复制交叉段
将父代 1 交叉点之间的基因段直接复制到子代 1 相同的位置，将父代 2 交叉点之间的基因段直接复制到子代 2 相同的位置。
3. 填充剩余部分
对于子代 1，从父代 2 中按顺序选取那些不在子代 1 交叉段中的基因，依次填充到子代 1 交叉段之前和之后的空位上；对于子代 2，从父代 1 中按顺序选取那些不在子代 2 交叉段中的基因，依次填充到子代 2 交叉段之前和之后的空位上。
示例说明
假设我们有两个父代个体：
父代 1: [1, 2, 3, 4, 5, 6, 7, 8, 9]
父代 2: [4, 1, 2, 8, 7, 6, 9, 3, 5]
步骤 1：随机选择交叉点
假设随机选择的交叉点为 2 和 5，那么交叉段就是索引 2 到 4 的位置（索引从 0 开始）。
步骤 2：复制交叉段
子代 1: [_, _, 3, 4, 5, _, _, _, _]
子代 2: [_, _, 2, 8, 7, _, _, _, _]
步骤 3：填充剩余部分
对于子代 1，从父代 2 中按顺序选取不在交叉段 [3, 4, 5] 中的基因 [4, 1, 2, 8, 7, 6, 9, 3, 5]，依次填充到空位上：
首先填充交叉段之前的空位，选取 4 （已在交叉段中，跳过），选取 1，填充到第一个空位；选取 2，填充到第二个空位。
然后填充交叉段之后的空位，选取 8，填充到第六个空位；选取 7，填充到第七个空位；选取 6，填充到第八个空位；选取 9，填充到第九个空位。
最终子代 1: [1, 2, 3, 4, 5, 8, 7, 6, 9]
对于子代 2，从父代 1 中按顺序选取不在交叉段 [2, 8, 7] 中的基因 [1, 2, 3, 4, 5, 6, 7, 8, 9]，依次填充到空位上：
首先填充交叉段之前的空位，选取 1，填充到第一个空位；选取 2 （已在交叉段中，跳过），选取 3，填充到第二个空位。
然后填充交叉段之后的空位，选取 4，填充到第六个空位；选取 5，填充到第七个空位；选取 6，填充到第八个空位；选取 9，填充到第九个空位。
最终子代 2: [1, 3, 2, 8, 7, 4, 5, 6, 9]
"""
