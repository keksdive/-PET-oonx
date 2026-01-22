import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ================= 配置区域 =================
# k=3 或 5 是论文常用的设置，计算速度快且对噪声有一定鲁棒性
KNN_K = 5
# 初始化一个全局分类器以节省开销
knn_classifier = KNeighborsClassifier(n_neighbors=KNN_K, n_jobs=-1)


def calculate_reward_supervised(selected_bands, new_action, X_train, y_train, X_val, y_val):
    """
    [核心修改] D3QN-SBS 监督式奖励函数
    使用 k-NN 在验证集上的准确率(OA)增量作为奖励。

    :param selected_bands: 当前已选的波段列表 (List[int])
    :param new_action: 智能体新选择的波段 (int)
    :param X_train:用于评估奖励的训练数据 (N_samples, N_total_bands)
    :param y_train:用于评估奖励的训练标签
    :param X_val:  用于评估奖励的验证数据
    :param y_val:  用于评估奖励的验证标签
    :return: reward (float) -> 准确率的提升值
    """

    # 1. 防止重复选择
    # 如果智能体选了已经存在的波段，给予惩罚 (或者奖励为0)
    if new_action in selected_bands:
        return -0.01  # 轻微惩罚，告诉它不要选重复的

    # 2. 构造新的波段组合
    # 注意：必须将 list 转换为 numpy 切片支持的格式
    current_set = selected_bands.copy()
    new_set = current_set + [new_action]

    # --- 计算旧状态的准确率 (Baseline) ---
    if len(current_set) == 0:
        old_acc = 0.0
    else:
        # 切片提取特征
        X_tr_old = X_train[:, current_set]
        X_val_old = X_val[:, current_set]

        knn_classifier.fit(X_tr_old, y_train)
        y_pred_old = knn_classifier.predict(X_val_old)
        old_acc = accuracy_score(y_val, y_pred_old)

    # --- 计算新状态的准确率 (New OA) ---
    X_tr_new = X_train[:, new_set]
    X_val_new = X_val[:, new_set]

    knn_classifier.fit(X_tr_new, y_train)
    y_pred_new = knn_classifier.predict(X_val_new)
    new_acc = accuracy_score(y_val, y_pred_new)

    # 3. 计算奖励 (增量形式)
    # 论文中 D3QN-SBS 的目标是最大化 OA，因此 Reward = ΔOA
    reward = new_acc - old_acc

    # [可选技巧] 放大奖励信号，帮助网络更快收敛
    # 如果提升很小（比如 0.001），网络可能学不动，可以乘一个系数
    # reward = reward * 10.0

    return reward