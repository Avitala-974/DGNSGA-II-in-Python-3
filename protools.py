def get_var_dict(upper, lower, genes, keys):
    """
    将基因编码映射为决策变量字典：
    keys —— 决策变量名称列表
    """
    var = {}
    for i, key in enumerate(keys):
        var[key] = genes[i]*(upper[i]-lower[i]) + lower[i]
    return var