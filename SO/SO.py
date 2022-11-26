"""
    蛇形优化算法的核心
"""
import math
import random

import numpy as np


def snake_optimization(search_agents_no, max_iter, chung_reynolds, dim, solution_bound):
    # 初始化
    # vec_flag表示的是等一下我们的c2和c3是取正还是取负
    vec_flag = [1, -1]
    # 有没有食物的阈值
    food_threshold = 0.25
    # 温度适不适合交配的阈值
    temp_threshold = 0.6
    # 常量c1,下面计算食物的质量的时候会用到
    c1 = 0.5
    # 模式阈值，在下面会使用到，当产生的随机值小于模式阈值就进入战斗模式，否则就进入交配模式
    model_threshold = 0.6
    # 常量c2,下面更新位置的时候会用到
    c2 = 0.5
    # 常量c3,用于战斗和交配
    c3 = 2
    # 生成种群
    X = solution_bound[0] + np.random.random_sample((search_agents_no, dim)) * (solution_bound[1] - solution_bound[0])
    # 将其转化成矩阵，主要的目的是为了和matlab中的步骤拟合，其实不转也可以，按照列表的方式计算也行
    X = np.matrix(X)
    # 个体的适应度
    fitness = [0 for i in range(search_agents_no)]
    # 开始计算适应度
    for i in range(search_agents_no):
        fitness[i] = chung_reynolds(X[i, :])
    # 计算出全局最佳适应度, 因为这个是第一次进行操作，不用和别的进行对比
    g_best = np.argmin(fitness)
    gy_best = fitness[g_best]
    # 得到食物的位置，其实就是当前全局最佳适应度的位置
    food = X[g_best, :]
    # 将种群进行分离,一半归为雌性，一半归为雄性
    male_number = int(np.round(search_agents_no / 2))
    female_number = search_agents_no - male_number
    male = X[0:male_number, :]
    female = X[male_number:search_agents_no, :]
    # 从总的适应度中分离出雄性的适应度
    male_individual_fitness = fitness[0:male_number]
    # 从总的适应度中分理处雌性的适应度
    female_individual_fitness = fitness[male_number:search_agents_no]
    # 计算雄性种群中的个体最佳
    male_fitness_best_index = np.argmin(male_individual_fitness)
    male_fitness_best_value = male_individual_fitness[male_fitness_best_index]
    # 对应的解决方案
    male_best_fitness_solution = male[male_fitness_best_index, :]
    # 计算雌性种群中的个体最佳
    female_fitness_best_index = np.argmin(female_individual_fitness)
    female_fitness_best_value = female_individual_fitness[female_fitness_best_index]
    # 对应的解决方案
    female_best_fitness_solution = female[male_fitness_best_index, :]
    # 更新位置之后的male
    new_male = np.matrix(np.zeros((male_number, dim)))
    # 更新位置之后的female
    new_female = np.matrix(np.zeros((female_number, dim)))
    # 记录每代最佳适应度
    gene_best_fitness = [0 for i in range(max_iter)]
    # 开始进行循环
    for t in range(max_iter):
        # 计算温度
        temp = math.exp(-(t / max_iter))
        # 计算食物的质量
        quantity = c1 * math.exp((t - max_iter) / max_iter)
        if quantity > 1:
            quantity = 1
        # 先判断食物的质量是不是超过了阈值
        if quantity < food_threshold:
            # 如果当前是没有食物的就寻找食物
            # 先是雄性
            for i in range(male_number):
                for j in range(dim):
                    # 先取得一个随机的个体
                    rand_leader_index = np.random.randint(0, male_number)
                    rand_male = male[rand_leader_index, :]
                    # 随机生成+或者是-,来判断当前的c2是取正还是负
                    negative_or_positive = np.random.randint(0, 2)
                    flag = vec_flag[negative_or_positive]
                    # 计算Am,np.spacing(1)是为了防止进行除法运算的时候出现除0操作
                    am = math.exp(
                        -(male_individual_fitness[rand_leader_index] / (male_individual_fitness[i] + np.spacing(1))))
                    new_male[i, j] = rand_male[0, j] + flag * c2 * am * (
                            (solution_bound[1] - solution_bound[0]) * random.random() + solution_bound[0])
            for i in range(female_number):
                for j in range(dim):
                    # 先取得一个随机的个体
                    rand_leader_index = np.random.randint(0, female_number)
                    rand_female = female[rand_leader_index, :]
                    # 随机生成+或者是-,来判断当前的c2是取正还是负
                    negative_or_positive = np.random.randint(0, 2)
                    flag = vec_flag[negative_or_positive]
                    # 计算Am,np.spacing(1)是为了防止进行除法运算的时候出现除0操作
                    am = math.exp(-(female_individual_fitness[rand_leader_index] / (
                            female_individual_fitness[i] + np.spacing(1))))
                    new_female[i, j] = rand_female[0, j] + flag * c2 * am * (
                            (solution_bound[1] - solution_bound[0]) * random.random() + solution_bound[0])
        else:
            # 当前有食物开始进入探索阶段
            # 先判断当前的温度是冷还是热
            if temp > temp_threshold:  # 表示当前是热的
                # 热了就不进行交配，开始向食物的位置进行移动
                # 雄性先移动
                for i in range(male_number):
                    # 随机生成+或者是-,来判断当前的c2是取正还是负
                    negative_or_positive = np.random.randint(0, 2)
                    flag = vec_flag[negative_or_positive]
                    for j in range(dim):
                        new_male[i, j] = food[0, j] + flag * c3 * temp * random.random() * (food[0, j] - male[i, j])
                # 更新雌性的位置
                for i in range(female_number):
                    # 随机生成+或者是-,来判断当前的c2是取正还是负
                    negative_or_positive = np.random.randint(0, 2)
                    flag = vec_flag[negative_or_positive]
                    for j in range(dim):
                        new_female[i, j] = food[0, j] + flag * c3 * temp * random.random() * (food[0, j] - female[i, j])
            else:
                # 如果当前的温度是比较的冷的，就比较适合战斗和交配
                # 生成一个随机值来决定是要战斗还是要交配
                model = random.random()
                if model < model_threshold:
                    # 当前进入战斗模式
                    # 更新雄性的位置
                    for i in range(male_number):
                        for j in range(dim):
                            # 先计算当前雄性的战斗的能力
                            fm = math.exp(-female_fitness_best_value / (male_individual_fitness[i] + np.spacing(1)))
                            new_male[i, j] = male[i, j] + c3 * fm * random.random() * (
                                    quantity * male_best_fitness_solution[0, j] - male[i, j])
                    # 更新雌性的位置
                    for i in range(female_number):
                        for j in range(dim):
                            # 先计算当前雌性的战斗的能力
                            ff = math.exp(-male_fitness_best_value / (female_individual_fitness[i] + np.spacing(1)))
                            new_female[i, j] = female[i, j] + c3 * ff * random.random() * (
                                    quantity * female_best_fitness_solution[0, j] - female[i, j])
                else:
                    # 当前将进入交配模式
                    # 雄性先交配
                    for i in range(male_number):
                        for j in range(dim):
                            # 计算当前雄性的交配的能力
                            mm = math.exp(-female_individual_fitness[i] / (male_individual_fitness[i] + np.spacing(1)))
                            new_male[i, j] = male[i, j] + c3 * random.random() * mm * (
                                    quantity * female[i, j] - male[i, j])
                    # 雌性先交配
                    for i in range(female_number):
                        for j in range(dim):
                            # 计算当前雄性的交配的能力
                            mf = math.exp(-male_individual_fitness[i] / (female_individual_fitness[i] + np.spacing(1)))
                            new_female[i, j] = female[i, j] + c3 * random.random() * mf * (
                                    quantity * male[i, j] - female[i, j])
                    # 产蛋
                    negative_or_positive = np.random.randint(0, 2)
                    egg = vec_flag[negative_or_positive]
                    if egg == 1:
                        # 拿到当前雄性种群中适应度最大的个体
                        male_best_fitness_index = np.argmax(male_individual_fitness)
                        new_male[male_best_fitness_index, :] = solution_bound[0] + random.random() * (
                                solution_bound[1] - solution_bound[0])
                        # 拿到当前雌性种群中适应度最大的
                        female_best_fitness_index = np.argmax(female_individual_fitness)
                        new_female[female_best_fitness_index, :] = solution_bound[0] + random.random() * (
                                solution_bound[1] - solution_bound[0])
        # 将更新后的位置进行处理
        # 处理雄性
        for j in range(male_number):
            # 如果当前更新后的值是否在规定的范围内
            flag_low = new_male[j, :] < solution_bound[0]
            flag_high = new_male[j, :] > solution_bound[1]
            new_male[j, :] = (np.multiply(new_male[j, :], ~(flag_low + flag_high))) + np.multiply(solution_bound[1], flag_high) +np.multiply(solution_bound[0], flag_low)
            # 计算雄性种群中每一个个体的适应度（这个是被更新过位置的）
            y = chung_reynolds(new_male[j, :])
            # 判断是否需要更改当前个体的历史最佳适应度
            if y < male_individual_fitness[j]:
                # 更新适应度
                male_individual_fitness[j] = y
                # 更新原有种群中个体的位置到新位置
                male[j, :] = new_male[j, :]
        # 得到雄性个体中的最佳适应度
        # 拿到索引
        male_current_best_fitness_index = np.argmin(male_individual_fitness)
        # 拿到值
        male_current_best_fitness = male_individual_fitness[male_current_best_fitness_index]
        # 处理雌性
        for j in range(female_number):
            # 如果当前更新后的值是否在规定的范围内
            flag_low = new_female[j, :] < solution_bound[0]
            flag_high = new_female[j, :] > solution_bound[1]
            new_female[j, :] = (np.multiply(new_female[j, :], ~(flag_low + flag_high))) + np.multiply(solution_bound[1], flag_high) +np.multiply(solution_bound[0], flag_low)
            # 计算雄性种群中每一个个体的适应度（这个是被更新过位置的）
            y = chung_reynolds(new_female[j, :])
            # 判断是否需要更改当前个体的历史最佳适应度
            if y < female_individual_fitness[j]:
                # 更新适应度
                female_individual_fitness[j] = y
                # 更新原有种群中个体的位置到新位置
                female[j, :] = new_female[j, :]
        # 得到雄性个体中的最佳适应度
        # 拿到索引
        female_current_best_fitness_index = np.argmin(female_individual_fitness)
        # 拿到值
        female_current_best_fitness = male_individual_fitness[female_current_best_fitness_index]
        # 判断是否需要更新雄性种群的全局最佳适应度
        if male_current_best_fitness < male_fitness_best_value:
            # 更新解决方案
            male_best_fitness_solution = male[male_current_best_fitness_index, :]
            # 更新最佳适应度
            male_fitness_best_value = male_current_best_fitness
        # 判断是否需要更新雌性种群的全局最佳适应度
        if female_current_best_fitness < female_fitness_best_value:
            # 更新解决方案
            female_best_fitness_solution = female[female_current_best_fitness_index, :]
            # 更新最佳适应度
            female_fitness_best_value = female_current_best_fitness
        if male_current_best_fitness < female_current_best_fitness:
            gene_best_fitness[t] = male_current_best_fitness
        else:
            gene_best_fitness[t] = female_current_best_fitness
        # 更新全局最佳适应度（这里是非常的奇怪的，不进行判断就直接更新了，他就能确定本代的最佳一定是比上一代好！！！）
        if male_fitness_best_value < female_fitness_best_value:
            gy_best = male_fitness_best_value
            # 更新食物的位置
            food = male_best_fitness_solution
        else:
            gy_best = female_fitness_best_value
            # 更新食物的位置
            food = female_best_fitness_solution
    return food, gy_best, gene_best_fitness
