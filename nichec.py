import random
import math

z = 1.0e-8
Nearzero = 1.0e-15

def get_MaxR(dimension, pointAmount, upper, lower):
    """
    计算初始拥挤半径R：
    依据均匀分布体积公式 R = 0.5*(prod(upper-lower)*2*dim/(N*pi))^(1/dim)
    """
    production = 1.0
    for i in range(dimension):
        production *= (upper[i] - lower[i])
    production *= 2*dimension
    R = 0.5 * pow(production/(pointAmount*math.pi), 1.0/dimension)
    return R

def reduce_radius(k, MaxK, genecount, R, upper, lower):
    production = 1.0
    for i in range(genecount):
        delta = upper[i] - lower[i]
        # 加保护，避免乘积变成0
        if delta < 1e-12:
            delta = 1e-12
        production *= delta

    # 取 d 次方根
    root_prod = production ** (1.0 / genecount)
    if root_prod < 1e-12:
        root_prod = 1e-12  # 防止除以0

    z1 = z / root_prod
    if z1 < 1e-12:
        z1 = 1e-12  # 保底

    C = R + z1
    ratio = C / z1
    if ratio <= 0.0:
        ratio = 1e-12  # 防止 log(0) 报错

    c = math.sqrt(abs(math.log(ratio)))
    if c == 0:
        c = 1e-6  # 防止除以0

    D = float(MaxK) / c
    q = float(k) / D
    f = C * math.exp(-q) - z1
    return max(0.0, f)


def caculate_nichecount(_pop, _S, _genCount, r, size):
    """
    计算共享拥挤度目标nicheCount：
    对合并种群两两计算欧氏距离，小于r时共享，累加shareF=1-d/r
    将结果附加到violation_objectives
    """
    sum_pop = _pop + _S
    nicheCount = [0.0] * len(sum_pop)

    for i in range(len(sum_pop)):
        for j in range(i + 1, len(sum_pop)):
            dist = math.sqrt(sum((sum_pop[i]['genes'][k] - sum_pop[j]['genes'][k]) ** 2 for k in range(_genCount)))
            if dist < r and dist > 1e-6:
                share = 1 - (dist / r)
                nicheCount[i] += share
                nicheCount[j] += share

    # 存储为独立属性
    for i, ind in enumerate(sum_pop):
        ind['niche_count'] = nicheCount[i]