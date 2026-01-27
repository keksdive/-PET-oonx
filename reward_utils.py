import numpy as np
from sklearn.feature_selection import mutual_info_classif


def calculate_hybrid_reward(selected_bands, new_band, X, y, alpha=1.0, beta=0.5):
    """
    计算混合奖励：互信息 (Relevance) - 冗余度 (Redundancy)

    参数:
    - selected_bands: 已选波段列表
    - new_band: 当前动作选择的新波段索引
    - X, y: 用于计算的采样数据 (建议采样2000-5000个即可，无需全量)
    - alpha: 互信息项的权重 (鼓励选择与标签强相关的波段)
    - beta: 冗余项的权重 (惩罚与已选波段相似的波段)
    """

    # 1. 计算互信息 (Relevance)
    # 衡量新波段包含多少关于标签的信息量
    # reshape(-1, 1) 是因为 mutual_info_classif 期望 2D 输入
    mi_score = mutual_info_classif(X[:, [new_band]], y, discrete_features=False, random_state=42)[0]

    # 2. 计算冗余度 (Redundancy)
    # 如果是第一个波段，没有冗余
    redundancy_score = 0.0
    if len(selected_bands) > 0:
        # 获取新波段向量
        vec_new = X[:, new_band]
        correlations = []
        for b in selected_bands:
            vec_existing = X[:, b]
            # 计算皮尔逊相关系数的绝对值
            corr = np.abs(np.corrcoef(vec_new, vec_existing)[0, 1])
            correlations.append(corr)

        # 取平均相关性或最大相关性作为惩罚
        redundancy_score = np.mean(correlations)

    # 3. 最终奖励公式
    # R = Relevance - Redundancy
    # 我们希望 MI 大 (正向)，Correlation 小 (负向)
    reward = alpha * mi_score - beta * redundancy_score

    return reward