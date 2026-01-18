import numpy as np
from sklearn.feature_selection import mutual_info_classif

def calculate_reward(selected_bands, new_action, entropies, mi_scores, alpha=0.5):
    """
    优化后的奖励函数
    :param entropies: 预计算的所有波段熵值
    :param mi_scores: 预计算的所有波段互信息得分 (波段与标签的相关性)
    :param alpha: 权重因子，平衡'信息量'与'判别力'
    """
    if len(selected_bands) == 0:
        old_val = 0
    else:
        # 旧状态得分：平均熵 + alpha * 平均互信息
        old_val = np.mean(entropies[selected_bands]) + alpha * np.mean(mi_scores[selected_bands])

    new_list = selected_bands + [new_action]
    # 新状态得分
    new_val = np.mean(entropies[new_list]) + alpha * np.mean(mi_scores[new_list])

    return new_val - old_val