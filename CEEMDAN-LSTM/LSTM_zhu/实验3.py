import numpy as np
import matplotlib.pyplot as plt

# 参数设置
citys = 15         # 城市数量（即DNA长度）
pc = 0.1           # 交叉概率
pm = 0.02          # 变异概率
popsize = 500      # 种群大小
iternum = 100      # 迭代次数

class GA(object):
    def __init__(self, citys, pc, pm, popsize):
        self.citys = citys       # 城市数量
        self.pc = pc             # 交叉概率
        self.pm = pm             # 变异概率
        self.popsize = popsize   # 种群大小

        # 创建初始种群，随机排列代表路径
        self.pop = np.vstack([np.random.permutation(citys) for _ in range(popsize)])

    def translateDNA(self, DNA, city_position):
        # 将DNA（路径表示）转换为路径的X和Y坐标
        lineX = np.empty_like(DNA, dtype=np.float64) # 用于存储X坐标的数组
        lineY = np.empty_like(DNA, dtype=np.float64) # 用于存储Y坐标的数组

        # 根据DNA中城市的顺序提取坐标
        for i, d in enumerate(DNA):
            city_coord = city_position[d]  # 按DNA顺序获取城市坐标
            lineX[i, :] = city_coord[:, 0]  # X坐标
            lineY[i, :] = city_coord[:, 1]  # Y坐标
        return lineX, lineY

    def getFitness(self, lineX, lineY):
        # 计算每条路径的总距离，并计算适应度
        totalDis = np.empty((lineX.shape[0],), dtype=np.float64)  # 存储每条路径的总距离
        for i, (xval, yval) in enumerate(zip(lineX, lineY)):
            # 通过城市间的欧几里得距离求和计算总距离
            totalDis[i] = np.sum(np.sqrt(np.square(np.diff(xval)) + np.square(np.diff(yval))))
        fitness = np.exp(self.citys * 2 / totalDis)  # 适应度与距离成反比的指数适应度函数
        return fitness, totalDis

    def selection(self, fitness):
        # 基于适应度的选择过程（轮盘赌选择）
        idx = np.random.choice(np.arange(self.popsize), size=self.popsize, replace=True, p=fitness/fitness.sum())
        return self.pop[idx]  # 根据计算出的概率返回选择的个体

    def crossover(self, parent, pop):
        # 交叉操作
        if np.random.rand() < self.pc:  # 根据概率决定是否进行交叉
            i = np.random.randint(0, self.popsize, size=1)  # 随机选择另一个个体
            cross_points = np.random.randint(0, 2, self.citys).astype(np.bool)  # 随机交叉点
            keep_city = parent[~cross_points]  # 保留第一个父母的城市
            swap_city = pop[i, np.isin(pop[i].ravel(), keep_city, invert=True)]  # 从第二个父母中交换城市
            parent[:] = np.concatenate((keep_city, swap_city))  # 用父母生成的新个体替换
        return parent  # 返回可能修改过的父母

    def mutation(self, child):
        # 变异操作
        for point in range(self.citys):
            if np.random.rand() < self.pm:  # 根据概率决定是否变异
                swap_point = np.random.randint(0, self.citys)  # 随机选择另一个点进行交换
                swapa, swapb = child[point], child[swap_point]  # 要交换的城市
                child[point], child[swap_point] = swapb, swapa  # 执行交换
        return child  # 返回可能变异过的子代

    def evolve(self, fitness):
        # 主要进化过程
        pop = self.selection(fitness)  # 根据适应度选择新种群
        pop_copy = pop.copy()  # 复制用于交叉操作
        for parent in pop:
            child = self.crossover(parent, pop_copy)  # 对父代进行交叉生成子代
            child = self.mutation(child)  # 对子代进行变异
            parent[:] = child  # 用新子代替换父代
        self.pop = pop  # 更新种群为新一代

class TSP(object):
    def __init__(self, citys):
        # 随机生成城市的位置
        self.city_position = np.random.rand(citys, 2)  # 在二维空间中的位置
        plt.ion()  # 启用绘制个体的交互模式

    def plotting(self, lx, ly, total_d):
        # 绘制当前最佳路径及其总距离
        plt.cla()  # 清空当前坐标轴
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')  # 绘制城市
        plt.plot(lx.T, ly.T, 'r-')  # 绘制路径
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})  # 显示总距离
        plt.xlim((-0.1, 1.1))  # 设置x轴范围
        plt.ylim((-0.1, 1.1))  # 设置y轴范围
        plt.pause(0.01)  # 暂停以更新绘图

if __name__ == '__main__':
    # 主执行入口
    ga = GA(citys=citys, pc=pc, pm=pm, popsize=popsize)  # 初始化遗传算法
    env = TSP(citys=citys)  # 初始化TSP环境

    # 进化循环
    for gen in range(iternum):
        lx, ly = ga.translateDNA(ga.pop, env.city_position)  # 将当前种群的DNA转为坐标
        fitness, total_distance = ga.getFitness(lx, ly)  # 计算适应度和总距离
        ga.evolve(fitness)  # 根据适应度进化种群
        best_idx = np.argmax(fitness)  # 获取最佳方案的索引
        print("Gen:", gen, " | best fit: %.2f" % fitness[best_idx])  # 打印当前代的最佳适应度

        env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])  # 绘制最佳路径

    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示最终绘图
