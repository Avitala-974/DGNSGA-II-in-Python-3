import random
import sys

def select_next_parent_population(_S, _pop, N):
    next_parent_pop = []
    C = []
    C.extend(_S)
    C.extend(_pop)

    # 使用带niche_count的非支配排序
    nondominated_rank = fast_non_dominated_sort(C, len(C))

    for front in nondominated_rank:
        if len(next_parent_pop) + len(front) <= N:
            next_parent_pop.extend(front)
        else:
            # 按拥挤度排序（包含niche_count）
            temp = crowding_distance(front)
            num_needed = N - len(next_parent_pop)
            next_parent_pop.extend(temp[:num_needed])
        if len(next_parent_pop) >= N:
            break
    return next_parent_pop[:N]


def fast_non_dominated_sort(pop, size):
    fronts = [[]]
    domination_counts = [0] * size
    dominated_solutions = [[] for _ in range(size)]

    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            if compare_indivial(pop[i], pop[j]):
                dominated_solutions[i].append(j)
            elif compare_indivial(pop[j], pop[i]):
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            fronts[0].append(pop[i])

    current_front = 0
    while len(fronts[current_front]) > 0:
        next_front = []
        for ind in fronts[current_front]:
            for dominated in dominated_solutions[pop.index(ind)]:
                domination_counts[dominated] -= 1
                if domination_counts[dominated] == 0:
                    next_front.append(pop[dominated])
        current_front += 1
        fronts.append(next_front)
    return fronts[:-1]


def compare_indivial(a, b):
    if a['efeasible'] != b['efeasible']:
        return a['efeasible'] == 1
    elif a['efeasible'] == 1:
        # 仅比较原始目标和违反目标
        for i in range(len(a['objectives'])):
            if a['objectives'][i] > b['objectives'][i]:
                return False
        for i in range(len(a['violation_objectives'])):
            if a['violation_objectives'][i] > b['violation_objectives'][i]:
                return False
        # 完全相同则不算支配
        if a['objectives'] == b['objectives'] and a['violation_objectives'] == b['violation_objectives']:
            return False
        return True
        # 目标相同比较违反目标
        a_vio = sum(a['violation_objectives'])
        b_vio = sum(b['violation_objectives'])
        return a_vio < b_vio
    else:
        return sum(a['violation_objectives']) < sum(b['violation_objectives'])


def crowding_distance(pop):
    for ind in pop:
        ind['distance'] = 0.0

    # 处理目标（假设最小化）
    for obj_idx in range(len(pop[0]['objectives'])):
        pop.sort(key=lambda x: x['objectives'][obj_idx], reverse=False)
        pop[0]['distance'] = float('inf')
        pop[-1]['distance'] = float('inf')
        if pop[-1]['objectives'][obj_idx] - pop[0]['objectives'][obj_idx] < 1e-4:
            continue
        for i in range(1, len(pop) - 1):
            pop[i]['distance'] += (pop[i + 1]['objectives'][obj_idx] - pop[i - 1]['objectives'][obj_idx])

    # 处理niche_count（最大化多样性）
    pop.sort(key=lambda x: x.get('niche_count', 0), reverse=True)
    pop[0]['distance'] = float('inf')
    pop[-1]['distance'] = float('inf')
    for i in range(1, len(pop) - 1):
        pop[i]['distance'] += (pop[i - 1]['niche_count'] - pop[i + 1]['niche_count'])

    pop.sort(key=lambda x: x['distance'], reverse=True)
    return pop