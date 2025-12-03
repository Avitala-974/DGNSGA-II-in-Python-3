import DCNSGA_II_DE_conf
import math
import random
import copy
import sys

def initialize_parent_population(size, genecount):
    """
    初始化父代种群：
    size —— 种群规模
    genecount —— 基因（决策变量）维度
    返回一个包含随机Genes和代数信息的个体列表
    """
    pop = []
    for i in range(size):
        ind = {}
        ind['extrainfo'] = {}
        # 基因取值在[0,1]随机初始化
        ind['genes'] = [random.random() for j in range(genecount)]
        ind['extrainfo']['generation'] = 0
        pop.append(ind)
    return pop

def caculate_pheno(pop, upper, lower, n, size):
    """
    将基因映射到实际决策变量（表型）：
    upper, lower —— 决策变量上下界向量
    n —— 基因维度，size —— 种群规模
    """
    for k in range(size):
        pop[k]['pheno'] = []
        for i in range(n):
            t = pop[k]['genes'][i] * (upper[i] - lower[i]) + lower[i]
            pop[k]['pheno'].append(t)

def evaluate_population(pop, evaluator, fill_result):
    """
    评估种群：调用用户定义的评估函数(evaluator)，
    并通过fill_result将结果填充回个体信息
    返回本次评估数量
    """
    results = []
    for i in range(len(pop)):
        pop[i]['id'] = i
        results.append(evaluator(pop[i]))
    fill_result(pop, results)
    return len(pop)

def get_fill_result(inds, rsts):
    """
    将评估结果填充回个体：
    rsts含'id','objectives','constraints','extrainfo.filename'
    执行可行性违反量转换
    """
    for rst in rsts:
        i = rst['id']
        ind = inds[i]
        ind['objectives'] = rst['objectives']
        # 转换违反量，大于0保留，否则置0
        ind['violations'] = [c if c>0 else 0 for c in rst['constraints']]
        ind['extrainfo']['filename'] = rst['extrainfo']['filename']
        ind['extrainfo']['constraints'] = rst['constraints']

def caculate_initial_max_violation(rsts):
    """
    计算初代最大违反量(maxG)：
    用于后续归一化违反目标，如果所有<1，则置1
    """
    maxG = [1 for _ in range(len(rsts[0]['violations']))]
    for rst in rsts:
        for k in range(len(rst['violations'])):
            if rst['violations'][k] > maxG[k]:
                maxG[k] = rst['violations'][k]
    return maxG

def caculate_violation_objective(maxG, rsts):
    """
    根据violations和maxG计算违反目标violation_objectives：
    = 1/m * sum(G(i)/maxG(i))
    """
    m = len(rsts[0]['violations'])
    for rst in rsts:
        vObj = sum(rst['violations'][h] / float(maxG[h]) for h in range(m))
        rst['violation_objectives'] = [vObj/m]

def mark_individual_efeasible(e, pop):
    """
    根据弹性边界e判断个体efeasible：
    如果所有violations<=e，则视为e可行
    """
    for ind in pop:
        ind['efeasible'] = int(all(ind['violations'][j] <= e[j] for j in range(len(e))))

def judge_population_efeasible(tmp):
    """
    判断整个种群是否都e可行：
    若全部efeasible，为1，否则0
    """
    return int(all(ind['efeasible']==1 for ind in tmp))

z = 1.0e-08
Nearzero = 1.0e-15

def reduce_boundary(eF, k, MaxK):
    """
    根据当前环境K收紧弹性边界：
    eF —— 初代maxG，k —— 当前环境计数，MaxK —— 最大环境变化数
    公式：f = eF[i]*exp(-(k/C)^2) (略去微小z项)
    """
    _e = []
    for val in eF:
        c = math.sqrt(math.log((val+z)/z))
        C = MaxK/c if c!=0 else Nearzero
        q = k/ C
        f = val * math.exp(-q*q)
        _e.append(max(0.0, f - z))
    return _e

# 从配置文件导入操作算子参数
CR = DCNSGA_II_DE_conf.CR
Pm = DCNSGA_II_DE_conf.Pm

def generate_offspring_population(n, _size, _tmp, _genCount):
    """
    基于DE算子生成子代：
    n      —— 当前代数
    _size  —— 种群规模
    _tmp   —— 父代列表
    _genCount —— 基因维度
    """
    S = []
    random.shuffle(_tmp)
    for i in range(_size):
        # 提取基因列表
        gene_pool = [ind['genes'] for ind in _tmp]
        offspring = create_offspring(i, gene_pool, _size, _genCount)
        # 按交叉率CR保留或替换基因
        for k in range(_genCount):
            if random.random() > CR:
                offspring['genes'][k] = _tmp[i]['genes'][k]
        # 按变异率Pm随机变异基因
        for k in range(_genCount):
            if random.random() <= Pm:
                offspring['genes'][k] = random.random()
        offspring['extrainfo'] = {'generation': n+1}
        S.append(offspring)
    return S

def create_offspring(n, ind, popSize, genCount):
    """
    DE变异算子：vi = pa + F*(pb - pc)
    选取三种群中(i+1,i+2,i+3)%popSize的个体作为pa,pb,pc
    对超出[0,1]值做对称映射
    """
    select = [ind[(n + i + 1)%popSize] for i in range(3)]
    F = random.uniform(0,1)
    offspring = {'genes': []}
    for i in range(genCount):
        temp = select[0][i] + F*(select[1][i] - select[2][i])
        # 对超出边界的值取对称
        if temp > 1:
            temp = 1.0 - (temp - int(temp))
        if temp < 0:
            temp = int(temp) - temp
        offspring['genes'].append(temp)
    return offspring

def get_efeasible_ratio(pop):
    """统计e可行个体比例"""
    total = len(pop)
    feasible = sum(1 for ind in pop if ind.get('efeasible', 0) == 1)
    ratio = feasible / total if total > 0 else 0
    print(f"[E-Feasible Ratio] 当前 e-可行个体数: {feasible}/{total} = {ratio:.2%}")
    return ratio

def repair(select, genCount, befF):
    """
    边界修复算子（备用）：确保vi生成值在[0,1]
    """
    Flist = [befF]
    for i in range(genCount):
        denom = select[2][i] - select[1][i]
        if denom == 0:
            Flist.append(befF)
        else:
            F1 = (1 - select[0][i]) / denom
            F2 = (0 - select[0][i]) / denom
            Flist.append(F1 if F1>=0 else F2)
    F = min(Flist)
    offgenes = []
    for i in range(genCount):
        temp = select[0][i] + F*(select[2][i] - select[1][i])
        temp = min(max(temp, 0.0), 1.0)
        offgenes.append(temp)
    return offgenes

import nichec
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
    """
    随环境演化收缩拥挤半径：
    C = R + z1，z1 = z/prod^(1/dim)，f=C*exp(-(k/D)^2)-z1
    """
    production = 1.0
    for i in xrange(genecount):
        production *= (upper[i] - lower[i])
    z1 = z/(production**(1.0/genecount))
    C = R + z1
    c = math.sqrt(abs(math.log(C/z1)))
    D = float(MaxK)/c
    q = float(k)/D
    f = C*math.exp(-q*q) - z1
    return max(0.0, f)

def caculate_nichecount(_pop, _S, _genCount, r, size):
    """
    计算共享拥挤度目标nicheCount：
    对合并种群两两计算欧氏距离，小于r时共享，累加shareF=1-d/r
    将结果附加到violation_objectives
    """
    sum_pop = _pop + _S
    nicheCount = [0.0]*size
    # 两两计算
    for i in range(size):
        for j in range(i+1, size):
            dist = math.sqrt(sum((sum_pop[i]['genes'][k]-sum_pop[j]['genes'][k])**2 for k in range(_genCount)))
            if dist < r:
                shareF = 1 - dist/r
                nicheCount[i] += shareF
                nicheCount[j] += shareF
    # 更新违反目标
    for i in range(size):
        vo = sum_pop[i]['violation_objectives']
        if len(vo) == 1:
            vo.append(nicheCount[i])
        else:
            vo[1] = nicheCount[i]