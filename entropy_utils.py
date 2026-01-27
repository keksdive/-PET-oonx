import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif
import time


def precompute_entropies(data):
    """
    计算每个波段的香农熵 (Shannon Entropy)
    目的：衡量波段的信息丰富程度（不论是否有用）。

    :param data: 形状为 (N_samples, N_bands) 的光谱数据
    :return: 形状为 (N_bands,) 的熵值数组
    """
    num_bands = data.shape[1]
    entropies = []

    print(f"📊 正在计算 {num_bands} 个波段的信息熵 (Information Quantity)...")
    start_time = time.time()

    for i in range(num_bands):
        band_pixels = data[:, i]
        # 计算直方图分布 (归一化为概率分布)
        # bins=100 既能保证精度，又不会过慢
        hist_counts, _ = np.histogram(band_pixels, bins=100, density=True)

        # 计算熵 (Base 2, 单位为 bit)
        # 加上 1e-10 防止 log(0)
        hist_counts = hist_counts[hist_counts > 0]
        band_ent = entropy(hist_counts, base=2)
        entropies.append(band_ent)

    print(f"✅ 熵计算完成，耗时: {time.time() - start_time:.2f}s")
    return np.array(entropies)


def precompute_mutual_information(data, labels):
    """
    [新增核心功能] 计算每个波段与标签的互信息 (Mutual Information)
    目的：衡量波段对 PET/非PET 分类的'判别力'。

    :param data: 形状为 (N_samples, N_bands) 的光谱数据
    :param labels: 形状为 (N_samples,) 的标签数据 (0或1)
    :return: 形状为 (N_bands,) 的互信息得分数组
    """
    print(f"🔍 正在计算波段与标签的互信息 (Discriminative Power)...")
    print("   (这可能需要几分钟，取决于数据量，请耐心等待)")
    start_time = time.time()

    # mutual_info_classif 专门处理分类任务
    # discrete_features=False 表示我们的光谱数据是连续数值
    # n_neighbors=3 是标准配置，计算 k-NN 熵估计
    # random_state=42 保证结果可复现
    mi_scores = mutual_info_classif(
        data,
        labels,
        discrete_features=False,
        n_neighbors=3,
        random_state=42,
        copy=False
    )

    print(f"✅ 互信息计算完成，耗时: {time.time() - start_time:.2f}s")

    # 归一化处理（可选）：将分数映射到 0-1 之间，方便与熵值加权
    if np.max(mi_scores) > 0:
        mi_scores = mi_scores / np.max(mi_scores)

    return mi_scores