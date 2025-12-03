import math
import numpy as np
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --------------------------
# 基础数据定义（与原始问题一致）
# --------------------------

# 省份列表
regions = [
    'Beijing', 'Tianjin', 'Hebei', 'Shanxi', 'Inner Mongolia', 'Liaoning',
    'Jilin', 'Heilongjiang', 'Shanghai', 'Jiangsu', 'Zhejiang', 'Anhui',
    'Fujian', 'Jiangxi', 'Shandong', 'Henan', 'Hubei', 'Hunan',
    'Guangdong', 'Guangxi', 'Hainan', 'Chongqing', 'Sichuan', 'Guizhou',
    'Yunnan', 'Shaanxi', 'Gansu', 'Qinghai', 'Ningxia', 'Xinjiang'
]

# 2022年各省基础数据
base_data = [
    {'coal': 91.05, 'oil': 774.92, 'gas': 197.95,
     'GDP_1': 111.5, 'GDP_2': 6605.1, 'GDP_3': 34894.3},  # Beijing
    {'coal': 3609.57, 'oil': 1660.88, 'gas': 127.52,
     'GDP_1': 273.1, 'GDP_2': 6038.9, 'GDP_3': 9999.3},  # Tianjin
    {'coal': 26612.52, 'oil': 2184.52, 'gas': 188.02,
     'GDP_1': 4410.3, 'GDP_2': 17050.1, 'GDP_3': 20910},  # Hebei
    {'coal': 60189.1, 'oil': 0, 'gas': 99.46,
     'GDP_1': 1340.4, 'GDP_2': 13840.8, 'GDP_3': 10461.3},  # Shanxi
    {'coal': 54967.75, 'oil': 344.84, 'gas': 90.56,
     'GDP_1': 2653.7, 'GDP_2': 11241.8, 'GDP_3': 9263.1},  # Inner Mongolia
    {'coal': 18420.78, 'oil': 9752.73, 'gas': 79,
     'GDP_1': 2597.6, 'GDP_2': 11755.8, 'GDP_3': 14621.7},  # Liaoning
    {'coal': 8117.48, 'oil': 953.76, 'gas': 37.14,
     'GDP_1': 1689.1, 'GDP_2': 4628.3, 'GDP_3': 6752.8},  # Jilin
    {'coal': 14184.66, 'oil': 1701.78, 'gas': 55.26,
     'GDP_1': 3609.8, 'GDP_2': 4648.9, 'GDP_3': 7642.2},  # Heilongjiang
    {'coal': 4641.22, 'oil': 2096.48, 'gas': 95,
     'GDP_1': 97, 'GDP_2': 11458.4, 'GDP_3': 33097.4},  # Shanghai
    {'coal': 27071.48, 'oil': 3991.61, 'gas': 305.34,
     'GDP_1': 4959.4, 'GDP_2': 55888.7, 'GDP_3': 62027.5},  # Jiangsu
    {'coal': 16553.01, 'oil': 6753.78, 'gas': 172.03,
     'GDP_1': 2324.8, 'GDP_2': 33205.2, 'GDP_3': 42185.4},  # Zhejiang
    {'coal': 18674.22, 'oil': 579.56, 'gas': 78.2,
     'GDP_1': 3513.7, 'GDP_2': 18588, 'GDP_3': 22943.3},  # Anhui
    {'coal': 9728.6, 'oil': 2580.98, 'gas': 55.41,
     'GDP_1': 3076.2, 'GDP_2': 25078.2, 'GDP_3': 24955.5},  # Fujian
    {'coal': 8477.43, 'oil': 719.55, 'gas': 39.03,
     'GDP_1': 2451.5, 'GDP_2': 14359.6, 'GDP_3': 15263.7},  # Jiangxi
    {'coal': 39010.36, 'oil': 13486.09, 'gas': 213.39,
     'GDP_1': 6298.6, 'GDP_2': 35014.2, 'GDP_3': 46122.3},  # Shandong
    {'coal': 21424.07, 'oil': 875.7, 'gas': 123.49,
     'GDP_1': 5817.8, 'GDP_2': 25465, 'GDP_3': 30062.2},  # Henan
    {'coal': 13126.78, 'oil': 1466.38, 'gas': 67.59,
     'GDP_1': 4986.7, 'GDP_2': 21240.6, 'GDP_3': 27507.6},  # Hubei
    {'coal': 9056.65, 'oil': 829.48, 'gas': 45.73,
     'GDP_1': 4602.7, 'GDP_2': 19182.6, 'GDP_3': 24885.1},  # Hunan
    {'coal': 19841.94, 'oil': 6634.04, 'gas': 295.69,
     'GDP_1': 5340.4, 'GDP_2': 52843.5, 'GDP_3': 70934.7},  # Guangdong
    {'coal': 9070.81, 'oil': 1594.58, 'gas': 33.22,
     'GDP_1': 4269.8, 'GDP_2': 8938.6, 'GDP_3': 13092.5},  # Guangxi
    {'coal': 1138.74, 'oil': 905.87, 'gas': 58.18,
     'GDP_1': 1417.8, 'GDP_2': 1310.9, 'GDP_3': 4089.5},  # Hainan
    {'coal': 5236.03, 'oil': 0, 'gas': 130.23,
     'GDP_1': 2012.1, 'GDP_2': 11693.9, 'GDP_3': 15423.1},  # Chongqing
    {'coal': 7806.56, 'oil': 1008.78, 'gas': 275.9,
     'GDP_1': 5964.3, 'GDP_2': 21157.1, 'GDP_3': 29628.4},  # Sichuan
    {'coal': 12219.99, 'oil': 0, 'gas': 53.93,
     'GDP_1': 2861.2, 'GDP_2': 7113, 'GDP_3': 10190.4},  # Guizhou
    {'coal': 8475.23, 'oil': 1003.31, 'gas': 23.93,
     'GDP_1': 4012.2, 'GDP_2': 10471.2, 'GDP_3': 14470.8},  # Yunnan
    {'coal': 24326.68, 'oil': 1925.92, 'gas': 147.95,
     'GDP_1': 2575.3, 'GDP_2': 15933.1, 'GDP_3': 14264.2},  # Shaanxi
    {'coal': 8486.45, 'oil': 1471.87, 'gas': 39.01,
     'GDP_1': 1515.3, 'GDP_2': 3945, 'GDP_3': 5741.2},  # Gansu
    {'coal': 1764.22, 'oil': 152.44, 'gas': 44.67,
     'GDP_1': 380.2, 'GDP_2': 1585.7, 'GDP_3': 1644.2},  # Qinghai
    {'coal': 16822.1, 'oil': 459.84, 'gas': 26.67,
     'GDP_1': 407.5, 'GDP_2': 2449.1, 'GDP_3': 2213},  # Ningxia
    {'coal': 29829.12, 'oil': 2529.3, 'gas': 156.63,
     'GDP_1': 2509.3, 'GDP_2': 7271.1, 'GDP_3': 7961}  # Xinjiang
]

# 各省GDP增长率约束
growth_rates = [
    {'2022-2025': 5, '2025-2030': 4.5},  # Beijing
    {'2022-2025': 5, '2025-2030': 4.5},  # Tianjin
    {'2022-2025': 5, '2025-2030': 4.5},  # Hebei
    {'2022-2025': 5, '2025-2030': 4.5},  # Shanxi
    {'2022-2025': 6, '2025-2030': 5.5},  # Inner Mongolia
    {'2022-2025': 5, '2025-2030': 4.5},  # Liaoning
    {'2022-2025': 5.5, '2025-2030': 5},  # Jilin
    {'2022-2025': 5, '2025-2030': 4.5},  # Heilongjiang
    {'2022-2025': 5, '2025-2030': 4.5},  # Shanghai
    {'2022-2025': 5, '2025-2030': 4.5},  # Jiangsu
    {'2022-2025': 5.5, '2025-2030': 5},  # Zhejiang
    {'2022-2025': 5.5, '2025-2030': 5},  # Anhui
    {'2022-2025': 5.5, '2025-2030': 5},  # Fujian
    {'2022-2025': 5, '2025-2030': 4.5},  # Jiangxi
    {'2022-2025': 5, '2025-2030': 4.5},  # Shandong
    {'2022-2025': 5.5, '2025-2030': 5},  # Henan
    {'2022-2025': 6, '2025-2030': 5.5},  # Hubei
    {'2022-2025': 5.5, '2025-2030': 5},  # Hunan
    {'2022-2025': 5, '2025-2030': 4.5},  # Guangdong
    {'2022-2025': 5, '2025-2030': 4.5},  # Guangxi
    {'2022-2025': 6, '2025-2030': 5.5},  # Hainan
    {'2022-2025': 6, '2025-2030': 5.5},  # Chongqing
    {'2022-2025': 5.5, '2025-2030': 5},  # Sichuan
    {'2022-2025': 5.5, '2025-2030': 5},  # Guizhou
    {'2022-2025': 5, '2025-2030': 4.5},  # Yunnan
    {'2022-2025': 5, '2025-2030': 4.5},  # Shaanxi
    {'2022-2025': 5.5, '2025-2030': 5},  # Gansu
    {'2022-2025': 4.5, '2025-2030': 4},  # Qinghai
    {'2022-2025': 5.5, '2025-2030': 5},  # Ningxia
    {'2022-2025': 6, '2025-2030': 5.5}  # Xinjiang
]

# 2025年全国约束
constraints_2025 = {
    'GDP_up': 1406756.962, 'GDP_low': 1363509.096,
    'ind1_up': 0.075, 'ind1_low': 0.066,  # 第一产业比例
    'ind2_up': 0.38, 'ind2_low': 0.3279,  # 第二产业比例
    'ind3_up': 0.6061, 'ind3_low': 0.52,  # 第三产业比例
    'Energy_up': 620000, 'Energy_low': 520000,
    'CO2_up': 1333990.051, 'CO2_low': 1200000,
    'coal_gr_up': 0.05, 'coal_gr_low': 0.00,  # 煤炭年增长率
    'oil_gr_up': 0.035, 'oil_gr_low': 0.015,  # 石油年增长率
    'gas_gr_up': 0.1, 'gas_gr_low': 0.06,  # 天然气年增长率
    'carbon_per_gdp_up': 0.9531571, 'carbon_per_gdp_low': 0.85,
    'energy_per_gdp_up': 0.422871401, 'energy_per_gdp_low': 0.38,
}

# 2030年全国约束
constraints_30 = {
    'GDP_up': 1749436.23, 'GDP_low': 1636210.915,
    'ind1_up': 0.07, 'ind1_low': 0.06,
    'ind2_up': 0.35, 'ind2_low': 0.27,
    'ind3_up': 0.67, 'ind3_low': 0.6,
    'Energy_up': 685000, 'Energy_low': 530000,
    'CO2_up': 1382054.622, 'CO2_low': 1210796.077,
    'coal_gr_up': 0.00, 'coal_gr_low': -0.01,  # 2025-2030年增长率
    'oil_gr_up': 0.03, 'oil_gr_low': -0.01,
    'gas_gr_up': 0.12, 'gas_gr_low': 0.04,
    'carbon_per_gdp_up': 0.79, 'carbon_per_gdp_low': 0.7,
    'energy_per_gdp_up': 0.418650184, 'energy_per_gdp_low': 0.30,
}


def _setup_bounds():
    """设置变量上下界（复用原始逻辑）"""
    n_var = 30 * 8 * 6
    xl = np.zeros(n_var)
    xu = np.zeros(n_var)

    for prov_idx in range(30):
        for year_idx in range(8):  # 2023-2030
            start = prov_idx * 48 + year_idx * 6  # 30省×8年×6变量=1440，每省48变量

            # 产业增长率边界（目标值±5%）
            if year_idx < 3:  # 2023-2025年
                target = growth_rates[prov_idx]['2022-2025'] / 100
            else:  # 2026-2030年
                target = growth_rates[prov_idx]['2025-2030'] / 100
            xl[start:start + 3] = [target - 0.05] * 3  # 下限
            xu[start:start + 3] = [target + 0.05] * 3  # 上限

            # 能源消耗边界（2022年值的60%-180%）
            coal_base = base_data[prov_idx]['coal']
            oil_base = base_data[prov_idx]['oil']
            gas_base = base_data[prov_idx]['gas']
            xl[start + 3] = coal_base * 0.6
            xu[start + 3] = coal_base * 1.3
            xl[start + 4] = oil_base * 0.6
            xu[start + 4] = oil_base * 1.5
            xl[start + 5] = gas_base * 0.6
            xu[start + 5] = gas_base * 1.8

    return xl, xu


def problem_initialize():
    """问题初始化配置：返回维度、上下界、约束数量、目标数量等"""
    dim = 30 * 8 * 6  # 变量维度：30省×8年×6变量=1440
    xl, xu = _setup_bounds()  # 获取上下界
    constraints_num = 56  # 约束总数
    objectives_num = 3  # 3个目标函数
    return (dim, xu.tolist(), xl.tolist(), None, constraints_num, objectives_num)


def calculate_results(x):
    """独立计算给定决策变量的结果，返回全国数据DataFrame"""
    # 初始化2022年各省数据
    prov_data = {i: {
        'gdp1': base_data[i]['GDP_1'],
        'gdp2': base_data[i]['GDP_2'],
        'gdp3': base_data[i]['GDP_3'],
        'coal': base_data[i]['coal'],
        'oil': base_data[i]['oil'],
        'gas': base_data[i]['gas']
    } for i in range(30)}

    # 结果存储结构
    yearly_national = {y: {'gdp1': 0, 'gdp2': 0, 'gdp3': 0, 'coal': 0, 'oil': 0, 'gas': 0}
                       for y in range(2023, 2031)}

    # 逐年计算
    for year_idx, year in enumerate(range(2023, 2031)):
        for prov_idx in range(30):
            # 提取变量
            start = prov_idx * 48 + year_idx * 6
            gr1, gr2, gr3 = x[start], x[start + 1], x[start + 2]
            coal, oil, gas = x[start + 3], x[start + 4], x[start + 5]

            # 更新各省GDP
            prov = prov_data[prov_idx]
            prov['gdp1'] *= (1 + gr1)
            prov['gdp2'] *= (1 + gr2)
            prov['gdp3'] *= (1 + gr3)

            # 累加全国数据
            y = yearly_national[year]
            y['gdp1'] += prov['gdp1']
            y['gdp2'] += prov['gdp2']
            y['gdp3'] += prov['gdp3']
            y['coal'] += coal
            y['oil'] += oil
            y['gas'] += gas

        # 计算全国能源、CO2及效率指标
        y = yearly_national[year]
        y['total_gdp'] = y['gdp1'] + y['gdp2'] + y['gdp3']
        y['energy_sum'] = (y['coal'] / 0.7143 + y['oil'] / 1.4286 + y['gas'] / 13.3) * 0.7539 - 34807
        y['co2'] = (y['coal'] * 19579492 + y['oil'] * 30303024 + y['gas'] * 216218880) / 10000000
        y['carbon_per_gdp'] = y['co2'] / y['total_gdp'] if y['total_gdp'] > 0 else 0
        y['energy_per_gdp'] = y['energy_sum'] / y['total_gdp'] if y['total_gdp'] > 0 else 0
        y['year'] = year

    return pd.DataFrame(list(yearly_national.values()))


def evaluate(ind):
    """评估函数：计算目标函数和约束"""
    x = ind['genes']  # 获取决策变量（1440维）
    try:
        # 计算全国数据
        national_df = calculate_results(x)

        # 计算累计目标值
        cum_gdp = national_df['total_gdp'].sum()
        cum_energy = national_df['energy_sum'].sum()
        cum_co2 = national_df['co2'].sum()

        # 计算约束违反值
        constraints = []

        # 1. 2025年全国约束（13项）
        y25 = national_df[national_df['year'] == 2025].iloc[0]
        constraints.extend([
            max(0, y25['total_gdp'] - constraints_2025['GDP_up']),  # GDP上限违反
            max(0, constraints_2025['GDP_low'] - y25['total_gdp']),  # GDP下限违反
            max(0, (y25['gdp1'] / y25['total_gdp']) - constraints_2025['ind1_up']),  # 一产比例上限
            max(0, constraints_2025['ind1_low'] - (y25['gdp1'] / y25['total_gdp'])),  # 一产比例下限
            max(0, (y25['gdp2'] / y25['total_gdp']) - constraints_2025['ind2_up']),  # 二产比例上限
            max(0, constraints_2025['ind2_low'] - (y25['gdp2'] / y25['total_gdp'])),  # 二产比例下限
            max(0, (y25['gdp3'] / y25['total_gdp']) - constraints_2025['ind3_up']),  # 三产比例上限
            max(0, constraints_2025['ind3_low'] - (y25['gdp3'] / y25['total_gdp'])),  # 三产比例下限
            max(0, y25['energy_sum'] - constraints_2025['Energy_up']),  # 能源上限
            max(0, constraints_2025['Energy_low'] - y25['energy_sum']),  # 能源下限
            max(0, y25['co2'] - constraints_2025['CO2_up']),  # CO2上限
            max(0, y25['carbon_per_gdp'] - constraints_2025['carbon_per_gdp_up']),  # 单位GDP碳排放上限
            max(0, constraints_2025['carbon_per_gdp_low'] - y25['carbon_per_gdp'])  # 单位GDP碳排放下限
        ])

        # 2. 2030年全国约束（13项）
        y30 = national_df[national_df['year'] == 2030].iloc[0]
        constraints.extend([
            max(0, y30['total_gdp'] - constraints_30['GDP_up']),
            max(0, constraints_30['GDP_low'] - y30['total_gdp']),
            max(0, (y30['gdp1'] / y30['total_gdp']) - constraints_30['ind1_up']),
            max(0, constraints_30['ind1_low'] - (y30['gdp1'] / y30['total_gdp'])),
            max(0, (y30['gdp2'] / y30['total_gdp']) - constraints_30['ind2_up']),
            max(0, constraints_30['ind2_low'] - (y30['gdp2'] / y30['total_gdp'])),
            max(0, (y30['gdp3'] / y30['total_gdp']) - constraints_30['ind3_up']),
            max(0, constraints_30['ind3_low'] - (y30['gdp3'] / y30['total_gdp'])),
            max(0, y30['energy_sum'] - constraints_30['Energy_up']),
            max(0, constraints_30['Energy_low'] - y30['energy_sum']),
            max(0, y30['co2'] - constraints_30['CO2_up']),
            max(0, y30['carbon_per_gdp'] - constraints_30['carbon_per_gdp_up']),
            max(0, constraints_30['carbon_per_gdp_low'] - y30['carbon_per_gdp'])
        ])

        # 3. 2022-2025能源增长率约束（6项）
        coal22 = sum(p['coal'] for p in base_data)
        oil22 = sum(p['oil'] for p in base_data)
        gas22 = sum(p['gas'] for p in base_data)
        coal25 = y25['coal']
        oil25 = y25['oil']
        gas25 = y25['gas']
        coal_gr = (coal25 / coal22) ** (1 / 3) - 1 if coal22 != 0 else 0
        oil_gr = (oil25 / oil22) ** (1 / 3) - 1 if oil22 != 0 else 0
        gas_gr = (gas25 / gas22) ** (1 / 3) - 1 if gas22 != 0 else 0
        constraints.extend([
            max(0, coal_gr - constraints_2025['coal_gr_up']),
            max(0, constraints_2025['coal_gr_low'] - coal_gr),
            max(0, oil_gr - constraints_2025['oil_gr_up']),
            max(0, constraints_2025['oil_gr_low'] - oil_gr),
            max(0, gas_gr - constraints_2025['gas_gr_up']),
            max(0, constraints_2025['gas_gr_low'] - gas_gr)
        ])

        # 4. 2025-2030能源增长率约束（6项）
        coal30 = y30['coal']
        oil30 = y30['oil']
        gas30 = y30['gas']
        coal_gr2 = (coal30 / coal25) ** (1 / 5) - 1 if coal25 != 0 else 0
        oil_gr2 = (oil30 / oil25) ** (1 / 5) - 1 if oil25 != 0 else 0
        gas_gr2 = (gas30 / gas25) ** (1 / 5) - 1 if gas25 != 0 else 0
        constraints.extend([
            max(0, coal_gr2 - constraints_30['coal_gr_up']),
            max(0, constraints_30['coal_gr_low'] - coal_gr2),
            max(0, oil_gr2 - constraints_30['oil_gr_up']),
            max(0, constraints_30['oil_gr_low'] - oil_gr2),
            max(0, gas_gr2 - constraints_30['gas_gr_up']),
            max(0, constraints_30['gas_gr_low'] - gas_gr2)
        ])

        # 5. 单位GDP能耗约束（4项）
        constraints.extend([
            max(0, y25['energy_per_gdp'] - constraints_2025['energy_per_gdp_up']),
            max(0, constraints_2025['energy_per_gdp_low'] - y25['energy_per_gdp']),
            max(0, y30['energy_per_gdp'] - constraints_30['energy_per_gdp_up']),
            max(0, constraints_30['energy_per_gdp_low'] - y30['energy_per_gdp'])
        ])

        # 6. 第三产业比例逐年上升（7项）
        for i in range(2023, 2030):
            y_curr = national_df[national_df['year'] == i].iloc[0]
            y_next = national_df[national_df['year'] == i + 1].iloc[0]
            ratio_curr = y_curr['gdp3'] / y_curr['total_gdp']
            ratio_next = y_next['gdp3'] / y_next['total_gdp']
            constraints.append(max(0, ratio_curr - ratio_next - 0.001))  # 允许±0.1%波动

        # 7. 单位GDP碳排放逐年下降（7项）
        for i in range(2023, 2030):
            y_curr = national_df[national_df['year'] == i].iloc[0]
            y_next = national_df[national_df['year'] == i + 1].iloc[0]
            carbon_curr = y_curr['carbon_per_gdp']
            carbon_next = y_next['carbon_per_gdp']
            constraints.append(max(0, carbon_next - carbon_curr - 0.001))  # 允许±0.1%波动

        # 应用惩罚（如果违反约束）
        penalty = 1e6 * sum(constraints)
        f1 = -cum_gdp + penalty  # 最大化GDP（转为最小化问题）
        f2 = cum_energy + penalty  # 最小化能源消耗
        f3 = cum_co2 + penalty  # 最小化CO2排放

        return {
            'id': ind['id'],  # 新增：添加个体ID
            'objectives': [f1, f2, f3],
            'constraints': constraints,
            'extrainfo': {
                'problem': 'EnergyOptimization',
                'filename': f"result_{ind['id']}.csv"  # 新增：补充filename字段
            }
        }

    except Exception as e:
        return {
            'id': ind.get('id', -1),  # 异常时的id默认值
            'objectives': [1e12, 1e12, 1e12],
            'constraints': [1e6] * 56,
            'extrainfo': {
                'problem': 'EnergyOptimization',
                'error': str(e),
                'filename': f"error_{ind.get('id', -1)}.csv"  # 异常时的filename
            }
        }