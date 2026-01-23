import numpy as np
import os
import json
import tensorflow as tf
from multi_agent import MultiAgentManager
from reward_utils import calculate_hybrid_reward
from visualization import visualize_spectral_curves

# ================= ⚙️ 配置区域 =================
DATA_DIR = r"D:\Processed_Result\67w-38w\procession-data"
NUM_BANDS_TO_SELECT = 30
TOTAL_EPISODES = 400  # 多智能体通常收敛更快

# 划分三个智能体的负责区域 (根据 208 个波段平均划分)
# VNIR (0-69), SWIR1 (70-139), SWIR2 (140-208)
AGENT_RANGES = [(0, 70), (70, 140), (140, 208)]
# ===============================================

gpus = tf.config.list_physical_devices('GPU')
if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)


def load_data():
    X = np.load(os.path.join(DATA_DIR, "X.npy"), mmap_mode='r')
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    return X, y


def prepare_balanced_data(X_full, y_full, n_per_class=2000):
    # (保持原有的平衡采样逻辑，此处省略以节省篇幅，直接调用即可)
    # 请复用您之前 main.py 中的 prepare_multiclass_drl_data 函数
    indices = []
    for cls in np.unique(y_full):
        idx = np.where(y_full == cls)[0]
        if len(idx) > 0:
            indices.append(np.random.choice(idx, min(len(idx), n_per_class), replace=False))
    indices = np.concatenate(indices)
    np.random.shuffle(indices)
    return X_full[indices].astype(np.float32), y_full[indices]


def train_multi_agent():
    # 1. 数据准备
    X_full, y_full = load_data()
    X_drl, y_drl = prepare_balanced_data(X_full, y_full, n_per_class=1500)
    total_bands = X_full.shape[1]

    # 2. 初始化多智能体系统
    manager = MultiAgentManager(total_bands, AGENT_RANGES)
    print(f"\n🚀 启动多智能体协作系统 ({len(manager.agents)} Agents)...")

    best_reward = -float('inf')
    best_bands = []

    for e in range(TOTAL_EPISODES):
        state = np.zeros(total_bands)
        selected_bands = []
        episode_reward = 0

        for step in range(NUM_BANDS_TO_SELECT):
            # A. 协同决策：Manager 询问所有 Agent 并选出最佳
            action = manager.get_global_action(state, selected_bands)

            # B. 计算混合奖励 (MI + Correlation)
            reward = calculate_hybrid_reward(selected_bands, action, X_drl, y_drl, alpha=2.5, beta=1.0)

            # C. 状态更新
            next_state = state.copy()
            next_state[action] = 1
            done = (len(selected_bands) == NUM_BANDS_TO_SELECT - 1)

            # D. 存储经验 (自动分发给对应的 Agent)
            manager.remember(state, action, reward, next_state, done)

            # E. 训练所有 Agent
            manager.train()

            state = next_state
            selected_bands.append(action)
            episode_reward += reward

        # F. 同步 Target 网络 & 衰减探索率
        manager.update_targets()
        manager.decay_epsilon()

        # G. 记录最佳结果
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_bands = sorted(selected_bands)
            print(f"🌟 [New Best] Ep {e + 1} | Reward: {episode_reward:.4f} | Bands: {best_bands}")

        if (e + 1) % 10 == 0:
            print(f"Ep {e + 1}/{TOTAL_EPISODES} | Reward: {episode_reward:.2f} | Epsilon: {manager.epsilon:.3f}")

    return best_bands


if __name__ == "__main__":
    final_bands = train_multi_agent()

    # 保存结果
    with open("best_bands_multi_agent.json", "w") as f:
        json.dump({"selected_bands": [int(b) for b in final_bands]}, f)

    print(f"\n✅ 最终选择波段: {final_bands}")

    # 可视化
    print("📊 生成多智能体选择分布图...")
    X_full, y_full = load_data()
    X_plot, y_plot = prepare_balanced_data(X_full, y_full, n_per_class=500)
    visualize_spectral_curves(X_plot, y_plot, final_bands, "Fig_MultiAgent_Selection.png")