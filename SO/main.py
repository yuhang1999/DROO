"""
    代码描述：蛇优化算算法的代码实现：
    作者：寒封csj
    参考文献：Snake Optimizer: A novel meta-heuristic optimization algorithm
    参考代码：https://ww2.mathworks.cn/matlabcentral/fileexchange/106465-snake-optimizer

"""
from FitnessFunction import chung_reynolds
from SO import snake_optimization
import matplotlib.pyplot as plt

# 定义维度
dim = 10
# 定义最大的迭代次数
max_iter = 1000
# 定义种群的的大小
search_agents_no = 30
# 定义边界
solution_bound = [0, 1]
food, global_fitness, gene_best_fitness = snake_optimization(search_agents_no, max_iter, chung_reynolds, dim, solution_bound)
plt.plot([i for i in range(max_iter)], gene_best_fitness)
print("最佳的解决方案：", food)
print("最佳适应度：", global_fitness)
plt.show()