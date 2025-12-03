import random

def problem_initialize():
    """问题初始化配置"""
    dim = 3  # 变量维度 (x, y, z)
    upper = [10.0, 10.0, 10.0]  # 变量上界
    lower = [0.0, 0.0, 0.0]  # 变量下界
    constraints_num = 4  # 约束数量
    objectives_num = 3  # 目标数量

    # 初始化种群（假设种群大小为100）
    population = []
    for _ in range(100):
        genes = [random.uniform(lower[i], upper[i]) for i in range(dim)]
        population.append({'genes': genes})

    return (dim, upper, lower, population, constraints_num, objectives_num)


def evaluate(ind):
    """评估函数：计算目标函数和约束"""
    x, y, z = ind['genes']
    # --------------------------
    # 1. 计算目标函数
    # --------------------------
    f1 = x ** 2 + y ** 2 + z ** 2  # 目标1：最小化各维度平方和
    f2 = (x - 3) ** 2 + (y - 5) ** 2 + z ** 2  # 目标2：最小化距离某个点的距离
    f3 = abs(x*y - z) + (x + y - z)**2   # 目标3：复杂非线性关系
    # -------------------------
    # 2. 计算约束违反值
    # --------------------------
    constraints = [
        max(0, x - 10),  # x <= 8
        max(0, y - 10),  # y <= 8
        max(0, z - 10),  # z <= 8
        max(0, 1 - y - z),
        max(0, x + y + z - 28),  # x + y + z <= 18
    ]
    # --------------------------
    # 3. 应用惩罚因子（如果违反约束）
    # --------------------------
    penalty = 1000 if any(c > 0 for c in constraints) else 0
    f1 += penalty
    f2 += penalty
    f3 += penalty
    return {
        'objectives': [f1, f2, f3],  # 目标函数值（含惩罚项）
        'constraints': constraints,  # 原始约束违反值
        'extrainfo': {'problem': 'c03'}
    }
