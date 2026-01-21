import numpy as np
import os
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 引用你的模块
from entropy_utils import precompute_entropies, precompute_mutual_information
from agent import BandSelectionAgent
from reward_utils import calculate_reward

# ================= 🔧 配置区域 =================
# [重要] 这里指向 save_data.py 生成的 .npy 文件夹
DATA_DIR = r"E:\SPEDATA\NP_newdata"

# [配置] 输出的波段数量
NUM_BANDS_TO_SELECT = 30

# [配置] 训练轮数
TOTAL_EPISODES = 300
ALPHA = 0.8  # 互信息权重

# ===============================================

# 显存配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU 显存按需分配已开启")
    except RuntimeError as e:
        print(e)


def load_cleaned_data_for_drl():
    """
    直接加载清洗后的 .npy 数据 (X.npy, y.npy)
    """
    print(f"🚀 [DRL] 正在加载清洗后的数据集: {DATA_DIR}")

    x_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"❌ 找不到 X.npy 或 y.npy，请先运行 save_data.py！路径: {DATA_DIR}")

    # 1. 加载数据
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)

    # 2. 检查数据
    # 我们不需要背景(0)，也不需要太多的样本导致计算太慢
    # save_data.py 生成的数据已经是纯净的材质数据了

    print(f"✅ 数据加载成功: {X.shape}")
    print(f"   材质标签分布: {np.unique(y, return_counts=True)}")

    # 3. 采样 (如果数据量太大，比如 > 5万，DRL计算互信息会很慢，建议采样)
    MAX_SAMPLES = 20000
    if X.shape[0] > MAX_SAMPLES:
        print(f"⚠️ 数据量过大 ({X.shape[0]}), 随机采样 {MAX_SAMPLES} 条用于特征选择...")
        indices = np.random.choice(X.shape[0], MAX_SAMPLES, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y


def train_dqn():
    # 1. 加载数据 (已清洗、已归一化)
    X_full, y_full = load_cleaned_data_for_drl()

    # ⚠️ 注意：因为 save_data.py 已经做了 Min-Max 归一化，
    # 这里不需要再做 SNV 或其他归一化，保持和训练时一致即可。
    # 如果你 save_data.py 没做归一化，这里才需要做。
    # 假设你用的是我刚才给的 save_data.py (含 Min-Max)，这里直接用。

    # 裁剪异常值 (Double check)
    X_full = np.clip(X_full, 0, 1)

    num_total_bands = X_full.shape[1]
    print(f"📊 总波段数: {num_total_bands}")

    # 2. 计算指标
    print("⚖️ 计算互信息 (Mutual Information)...")
    # 这里的 y_full 包含 1(PET), 2(CC), 3(PA) 等
    # 互信息会自动计算波段与这些类别的相关性
    mi_scores = precompute_mutual_information(X_full, y_full)

    print("📉 计算熵 (Entropy)...")
    entropies = precompute_entropies(X_full)

    # 归一化指标到 0-1
    mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-6)
    entropies = (entropies - np.min(entropies)) / (np.max(entropies) - np.min(entropies) + 1e-6)

    # 3. 训练 Agent
    agent = BandSelectionAgent(num_total_bands)

    print(f"\n🔥 开始筛选特征波段 (目标: {NUM_BANDS_TO_SELECT}个)...")

    best_reward = -float('inf')
    best_bands = []

    for e in range(TOTAL_EPISODES):
        state = np.zeros(num_total_bands)
        selected_bands = []
        episode_reward = 0

        for step in range(NUM_BANDS_TO_SELECT):
            action = agent.get_action(state, selected_bands)

            # 计算奖励
            reward = calculate_reward(selected_bands, action, entropies, mi_scores, alpha=ALPHA)

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

        if (e + 1) % 10 == 0:
            print(f"Episode {e + 1}/{TOTAL_EPISODES} | Reward: {episode_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

    print(f"\n🏆 筛选完成。共筛选 {len(best_bands)} 个材质特征波段:\n{best_bands}")
    return best_bands


if __name__ == "__main__":
    final_bands = train_dqn()

    if not final_bands:
        print("⚠️ 筛选失败，使用默认波段")
        final_bands = list(range(30))

    output_filename = "best_bands_config.json"
    save_data = {
        "description": "Selected using Cleaned Normalized Data (X.npy)",
        "count": len(final_bands),
        "selected_bands": [int(b) for b in final_bands]
    }

    with open(output_filename, "w") as f:
        json.dump(save_data, f, indent=4)

    print(f"💾 配置文件已更新: {os.path.abspath(output_filename)}")