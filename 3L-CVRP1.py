import numpy as np
import pandas as pd
import copy
import random
import matplotlib.pyplot as plt


distmartix = pd.read_excel("distmartix.xlsx", header=None, index_col=None)  # 距离矩阵
init_d = distmartix.values[0].sum() * 2  # 最大距离为每个工作台往返原点的距离和
order = pd.read_excel("order.xlsx")  # 订单计划
workplace_num = order["workplace"].nunique()  # 获取工作台数
workplace_id = order["workplace"].unique()  # 获取工作台的id
material = pd.read_excel("material.xlsx")  # 原料信息
material["material"] = material.index
material_num = material["material"].nunique()  # 原料数
order = order.merge(material, on="material", how='left')  # 合并两个表
order["n*M"] = order["num"] * order["M"]  # 计算每个订单所需物品的总质量
order["n*V"] = order["num"] * order["V"]  # 计算每个订单所需物品的总体积
workplace_M = order.groupby("workplace")["n*M"].sum()  # 计算每个工作台所需的物品总质量
workplace_V = order.groupby("workplace")["n*V"].sum()  # 计算每个工作台所需的物品体积
max_trollyid = 4  # 允许使用的最大小车数


class Box():
    def __init__(self, l, w, h):
        """创建一个箱子"""
        self.max_l = l
        self.max_w = w
        self.max_h = h

    def IsFeasible(self, set_point, item_info):
        """判断该放置点是否可以放入箱子"""
        x, y, z = set_point[0], set_point[1], set_point[2]
        l, w, h = item_info[0], item_info[1], item_info[2]
        if x + l > self.max_l or y + w > self.max_w or z + h > self.max_h:  # 判断是否超出箱子的长宽高
            return False
        else:
            return True

    def GetTask(self, task):
        """判断该装箱方案是否能成功"""
        set_points = [0, 0, 0]  # 可选放置点，初始为（0，0，0）
        lx, lz = 0, 0  # 参考线
        for i in range(len(task)):
            for j in range(len(task[i])):  # 遍历装箱任务中的每一个子任务
                item_info = task[i][j]
                item_l, item_w, item_h = item_info[0], item_info[1], item_info[2]  # 获取当前的物品长宽高
                if item_l > self.max_l or item_w > self.max_w or item_h > self.max_h:  # 如果该物品长宽高比盒子大则返回False
                    return False
                if self.IsFeasible(set_points, item_info) is False: # 如果该放置点不能装
                    set_points[1] = 0  # 放置点向z轴移动
                    set_points[2] += lz  
                    lz = 0
                    if self.IsFeasible(set_points, item_info) is False:  # 如果还不能装 
                        set_points[1] = 0  # 放置点向x轴移动
                        set_points[2] = 0
                        set_points[0] += lx
                        lx = 0
                        if self.IsFeasible(set_points, item_info) is False:
                            return False
                set_points[1] += item_info[1] 
                lx, lz = max(lx, item_info[0]), max(lz, item_info[2])
        return True

    def SA(self, task, alpha=0.95, min_t=0.1):
        """模拟退火算法求解装箱"""
        
        def ChangeSequence(task):
            """第一类邻域选择，随机交换两个物品"""
            change_index = np.random.randint(0, len(task))
            if len(task[change_index]) > 1:
                item1, item2 = random.sample(list(range(len(task[change_index]))), 2)
                task[change_index][item1], task[change_index][item2] = task[change_index][item2], task[change_index][item1]

        def ChangeDirection(task):
            """第二类邻域选择，随机交换某个物品的方向"""
            change_index = np.random.randint(0, len(task))
            change_item = np.random.randint(0, len(task[change_index]))
            task[change_index][change_item][0], task[change_index][change_item][1] = \
                task[change_index][change_item][1], task[change_index][change_item][0]

        t = 1
        while t > min_t:
            if np.random.uniform(0, 1) > 0.5:
                ChangeDirection(task)
            else:
                ChangeSequence(task)
            if self.GetTask(task):
                return True
            t *= alpha
        return False


class Trolly():
    max_g = 150  # 负重
    max_l = 100  # 长
    max_w = 50  # 宽
    max_h = 60  # 高
    max_v = max_l * max_w * max_h  # 体积
    g_limit = {"l": (10, 90), "w": (10, 40), "h": (0, 60)}  # 重心位置
    init_box = Box(max_l, max_w, max_h)  # 标准车厢空间

    def __init__(self, index):
        self.id = int(index)  # 小车编号
        self.g = 0  # 小车负重
        self.v = 0  # 承载原料体积
        self.travel = []  # 小车路线
        self.dist = 0  # 小车路线距离
        self.gravity_center = [50, 25, 30]  # 初始重心为小车中心
        self.box = copy.deepcopy(Trolly.init_box)  # 每辆小车的具体空间

    def IsOverWeight(self, task):
        """判断是否超重"""
        for i in range(len(task)):
            self.g += workplace_M.loc[task[i]]
            if self.g > Trolly.max_g:
                self.g = 0
                return True
        return False

    def IsOverVolume(self, task):
        """判断是否超过体积"""
        for i in range(len(task)):
            self.v += workplace_V.loc[task[i]]
            if self.v > Trolly.max_v:
                self.v = 0
                return True
        return False

    def IsOverPack(self, task):
        """判断是否能装箱"""

        def GetPackTask(task):
            """将任务转化为装箱任务"""
            pack_task = []
            for i in range(len(task)):
                workplace_task = []
                need_num = order[order["workplace"] == task[i]]["num"].values.tolist()
                need_material_info = order[order["workplace"] == task[i]][["L", "B", "H"]].values.tolist()
                for j in range(len(need_material_info)):
                    workplace_task += [need_material_info[j]] * need_num[j]
                pack_task.append(workplace_task)
            return pack_task

        pack_task = GetPackTask(task)
        if self.box.SA(pack_task):
            return False
        else:
            return True

    def IsOverGravityCenter(self):
        """判断重心是否超出限制"""
        return False

    def CalDist(self):
        """计算小车的路线距离"""
        if self.travel:
            self.dist += distmartix.loc[0][self.travel[0]] + distmartix.loc[0][self.travel[-1]]
            for i in range(len(self.travel) - 1):
                self.dist += distmartix.loc[self.travel[i]][self.travel[i + 1]]

    def GetTask(self, task):
        """判断小车能否完成任务"""
        if task:
            if self.IsOverWeight(task):  # 先判断是否超重
                return False
            if self.IsOverVolume(task):  # 判断是否超过体积
                return False
            if self.IsOverPack(task):  # 是否能装箱
                return False
            #if self.IsOverGravityCenter():  # 是否超出重心限制
            #    return False
            self.travel = task
            self.CalDist()
            return True
        return False


class Plan():
    def __init__(self, chromosome):
        self.routes = self.transcoding(chromosome) # 获取路线
        self.trollys = self.IsFeasible()  # 计划中每个小车的情况
        self.score = self.objectfun()  # 计划得分

    def transcoding(self, chromosome):
        """将染色体转码成计划"""
        plans = {}
        introlly = chromosome[: workplace_num]
        priority = chromosome[workplace_num: 2 * workplace_num]
        for i in range(workplace_num):
            if plans.get(introlly[i], -1) == -1:
                plans[introlly[i]] = [[priority[i], i + 1]]
            else:
                plans[introlly[i]].append([priority[i], i + 1])
        routes = {}
        for trolly_id in plans.keys():
            plans[trolly_id].sort(reverse=True)
            routes[trolly_id] = []
            for i in range(len(plans[trolly_id])):
                routes[trolly_id].append(plans[trolly_id][i][1])
        return routes

    def IsFeasible(self):
        """判断计划是否可行，可行则返回小车集合"""
        trollys = []
        for trolly_id, trolly_task in self.routes.items():
            trolly = Trolly(trolly_id)
            if trolly.GetTask(trolly_task):
                trollys.append(trolly)
            else:
                return []
        return trollys

    def objectfun(self):
        """计算方案的得分"""
        if self.trollys:
            total_d = 0
            total_v = 0
            for trolly in self.trollys:
                total_d += trolly.dist
                total_v += trolly.v / Trolly.max_v
            return 1 - total_d / init_d + total_v / len(self.trollys)
        return 0

    def info(self):
        print("计划信息：")
        for i in range(len(self.trollys)):
            print("小车{}号: 路线{}, 距离{}, 利用率{}".format(self.trollys[i].id, self.trollys[i].travel, self.trollys[i].dist,
                                                    round(self.trollys[i].v/Trolly.max_v, 4)))
        print("计划得分为: ", round(self.score, 4))


class GA():
    def __init__(self, population_size, generation, pc, pm):
        self.population_size = population_size  # 种群数
        self.generation = generation  # 迭代次数
        self.pc = pc  # 交叉概率
        self.pm = pm  # 变异概率
        self.population, self.population_fitness = self.InitialPopulation()  # 种群集合 种群得分

    def InitialPopulation(self):
        """初始化可行种群"""
        init_population = []
        init_fitness = []
        count = 0
        while count < self.population_size:
            chromosome = np.array([])
            first = np.random.randint(1, max_trollyid+1, (workplace_num, ))
            second = np.random.randint(1, 10, (workplace_num, ))
            chromosome = np.concatenate([chromosome, first])
            chromosome = np.concatenate([chromosome, second])
            plan = Plan(chromosome)
            if plan.trollys:
                init_population.append(chromosome)
                init_fitness.append(plan.score)
                count += 1
        return init_population, np.array(init_fitness)

    def BestChromosome(self):
        """种群中最好的个体以及得分"""
        max_id = int(np.argmax(self.population_fitness))
        return self.population[max_id], self.population_fitness[max_id]

    def select(self):
        """选择算子"""
        next_population = []
        p = self.population_fitness / self.population_fitness.sum()
        q = np.cumsum(p)
        e = np.random.uniform(0, 1, (self.population_size,))
        for i in range(self.population_size):
            for j in range(self.population_size):
                if e[i] < q[j]:
                    next_population.append(copy.deepcopy(self.population[j]))
                    break
        return next_population

    def cross(self, next_population):
        """交叉操作"""
        chromosome_len = len(next_population[0])
        for i in range(1, self.population_size, 2):
            if np.random.uniform(0, 1) < self.pc:
                cpoint = np.random.randint(1, chromosome_len-1)
                for j in range(cpoint, chromosome_len):
                    next_population[i][j], next_population[i-1][j] = next_population[i-1][j], next_population[i][j]

    def variation(self, next_population):
        """变异操作 (变异操作是针对每一个基因点)"""
        chromosome_len = len(next_population[0])
        for i in range(self.population_size):
            for j in range(chromosome_len):
                if np.random.uniform(0, 1) < self.pm:
                    if j < workplace_num:
                        next_population[i][j] = np.random.randint(1, max_trollyid+1)
                    elif j < workplace_num * 2:
                        next_population[i][j] = np.random.randint(1, 10)

    def change(self, next_population):
        """更新种群, 将后一代替换前一代"""
        next_population_fitness = np.zeros((self.population_size, ))
        for i in range(self.population_size):
            plan = Plan(next_population[i])
            next_population_fitness[i] = plan.score
        min_id = int(np.argmin(next_population_fitness))
        max_id = int(np.argmax(self.population_fitness))
        next_population[min_id] = copy.deepcopy(self.population[max_id])
        next_population_fitness[min_id] = self.population_fitness[max_id]
        self.population = next_population
        self.population_fitness = next_population_fitness

    def main(self):
        scores = []
        for i in range(self.generation):
            best_chromosome, best_score = self.BestChromosome()
            scores.append(best_score)
            next_population = self.select()
            self.cross(next_population)
            self.variation(next_population)
            self.change(next_population)
        return scores


if __name__ == '__main__':
    ga = GA(10, 20, 0.9, 0.05)
    y = ga.main()
    x = list(range(len(y)))
    plt.plot(x, y)
    plt.show()
    best_chromosome, best_score = ga.BestChromosome()
    print (best_chromosome, best_score)
    best_plan = Plan(best_chromosome)
    best_plan.info()


