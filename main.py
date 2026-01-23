import numpy as np
import os
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from multi_agent import BandSelectionAgent
from reward_utils import calculate_hybrid_reward  # <--- 引用改进后的奖励
from visualization import visualize_spectral_curves  # <--- 引用改进后的绘图
import datetime

# ================= 🔧 配置区域 =================
DATA_DIR = r"D:\Processed_Result\67w-38w\procession-data"  # 指向 save_data.py 输出的目录
NUM_BANDS_TO_SELECT = 30
TOTAL_EPISODES = 500  # 使用 MI 后收敛更快，可以适当减少轮数

# DRL 专用数据集大小 (每类样本数)
# MI 计算比 k-NN 快，但为了稳健，保持适中大小
SAMPLES_PER_CLASS = 2000
# ===============================================

# 显存配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass


def load_data():
    x_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    if not os.path.exists(x_path): raise Exception(f"Data not found in {DATA_DIR}")
    X = np.load(x_path, mmap_mode='r')
    y = np.load(y_path)
    return X, y


def prepare_multiclass_drl_data(X_full, y_full, samples_per_class=2000):
    """
    [改进] 支持多类别 (0, 1, 2) 的平衡采样
    """
    print(f"⚖️ 正在平衡多类别数据集 (目标: 每类 {samples_per_class} 个)...")

    unique_classes = np.unique(y_full)
    selected_indices = []

    for cls in unique_classes:
        idx_cls = np.where(y_full == cls)[0]
        count = len(idx_cls)
        print(f"   - Class {cls} 原始数量: {count}")

        if count > 0:
            n_select = min(count, samples_per_class)
            selected = np.random.choice(idx_cls, n_select, replace=False)
            selected_indices.append(selected)

    # 合并所有类别的索引
    selected_indices = np.concatenate(selected_indices)
    np.random.shuffle(selected_indices)

    # 加载数据到内存
    X_balanced = X_full[selected_indices].astype(np.float32)
    y_balanced = y_full[selected_indices].astype(np.float32)  # MI计算可能需要转int，但sklearn支持float标签作为分类

    print(f"✅ 平衡完成: 总样本数 {len(y_balanced)}")
    return X_balanced, y_balanced


def train_dqn():
    # 1. 加载数据
    X_full, y_full = load_data()
    num_total_bands = X_full.shape[1]

    # 2. 获取平衡数据集 (包含 PET, PA, Others)
    X_drl, y_drl = prepare_multiclass_drl_data(X_full, y_full, SAMPLES_PER_CLASS)

    # 3. 初始化 Agent
    agent = BandSelectionAgent(num_total_bands)
    print(f"\n🔥 开始训练 (Hybrid Reward: MI + Correlation)...")

    best_reward = -float('inf')
    best_bands = []

    for e in range(TOTAL_EPISODES):
        state = np.zeros(num_total_bands)
        selected_bands = []
        episode_reward = 0

        for step in range(NUM_BANDS_TO_SELECT):
            action = agent.get_action(state, selected_bands)

            # === [核心修改] 使用混合奖励函数 ===
            # alpha=2.0 加大相关性权重，beta=1.0 抑制冗余
            reward = calculate_hybrid_reward(selected_bands, action, X_drl, y_drl, alpha=2.0, beta=1.0)

            # 记录/更新状态
            next_state = state.copy()
            next_state[action] = 1
            done = (len(selected_bands) == NUM_BANDS_TO_SELECT - 1)

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            selected_bands.append(action)
            episode_reward += reward

        agent.update_target_network()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_bands = sorted(selected_bands)
            print(f"🌟 [New Best] Ep {e + 1} | Reward: {episode_reward:.4f} | Bands: {best_bands}")

        if (e + 1) % 10 == 0:
            print(f"Ep {e + 1}/{TOTAL_EPISODES} | R: {episode_reward:.2f} | Eps: {agent.epsilon:.3f}")

    print(f"\n🏆 最终筛选结果: {best_bands}")
    return best_bands


if __name__ == "__main__":
    # 1. 训练与选择
    final_bands = train_dqn()

    # 2. 保存配置
    with open("best_bands_mi.json", "w") as f:
        json.dump({"selected_bands": [int(b) for b in final_bands]}, f)

    # 3. [核心修改] 执行可视化验证
    print("\n📊 正在生成论文级可视化图表...")
    X_full, y_full = load_data()
    # 采样少量数据用于绘图 (避免绘图太慢)
    X_plot, y_plot = prepare_multiclass_drl_data(X_full, y_full, samples_per_class=500)

    visualize_spectral_curves(
        X_plot,
        y_plot,
        selected_bands=final_bands,
        save_path="Fig10_Spectral_Selection.png"
    )