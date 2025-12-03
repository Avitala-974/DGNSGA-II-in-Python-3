def problem_initialize():
    """问题初始化配置"""
    dim = 2  # 变量维度 (x和y)
    upper = [3.0, 3.0]  # 变量上界
    lower = [0.0, 0.0]  # 变量下界shuju
    constraints_num = 4  # 约束数量 (x和y的上下界)
    objectives_num = 2  # 目标数量
    return (dim, upper, lower, None, constraints_num, objectives_num)


def evaluate(ind):
    """评估函数：计算目标函数和约束"""
    x, y = ind['genes']
    # --------------------------
    # 1. 计算目标函数
    # --------------------------
    f1 = x ** 2 + y ** 2
    f2 = (x - 2) ** 2 + (y - 2) ** 2
    # --------------------------
    # 2. 计算约束违反值
    # --------------------------
    constraints = [
        max(0, 1 - x ),  # x >= 0 的违反程度
        max(0, x - 3),  # x <= 4 的违反程度
        max(0, 1 -y),  # y >= 0 的违反程度
        max(0, y - 3)  # y <= 4 的违反程度
    ]
    # --------------------------
    # 3. 应用惩罚因子（如果违反约束）
    # --------------------------
    if any(c > 0 for c in constraints):
        penalty = 1000
        f1 += penalty
        f2 += penalty

    return {
        'objectives': [f1, f2],  # 目标函数值（含惩罚项）
        'constraints': constraints,  # 原始约束违反值
        'extrainfo': {'problem': 'C02'}
    }