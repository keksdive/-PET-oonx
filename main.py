import numpy as np
import os
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from agent import BandSelectionAgent
from reward_utils import calculate_reward_supervised  # 确保这里引用的是修改后的 k-NN 版本
import datetime

# ================= 🔧 配置区域 =================
DATA_DIR = r"E:\SPEDATA\NP_new1.0.2"  # 指向你新生成的数据路径
NUM_BANDS_TO_SELECT = 30
TOTAL_EPISODES = 500

# 自动保存路径
AUTO_SAVE_DIR = r"D:\best-bands"
if not os.path.exists(AUTO_SAVE_DIR):
    os.makedirs(AUTO_SAVE_DIR)

# DRL 专用数据集大小 (每类样本数)
# 建议：每类 2500，总共 5000。太大会导致 k-NN 计算奖励变慢。
SAMPLES_PER_CLASS = 2500
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

    # 使用 mmap_mode='r' 可以避免一次性把 40万数据读入内存，节省内存
    X = np.load(x_path, mmap_mode='r')
    y = np.load(y_path)
    return X, y


def prepare_balanced_drl_data(X_full, y_full, samples_per_class=2000):
    """
    [新增] 构造一个严格平衡的 (1:1) 小规模数据集用于 DRL 奖励计算
    """
    print(f"⚖️ 正在平衡数据集 (目标: 每类 {samples_per_class} 个)...")

    # 1. 找出正负样本索引
    idx_pos = np.where(y_full == 1)[0]
    idx_neg = np.where(y_full == 0)[0]

    print(f"   - 原始正样本数: {len(idx_pos)}")
    print(f"   - 原始负样本数: {len(idx_neg)}")

    # 2. 检查数据量是否足够
    real_samples = min(len(idx_pos), len(idx_neg), samples_per_class)

    # 3. 随机抽取 (无放回)
    # 注意：因为 X 是 mmap，这里只操作索引
    selected_pos = np.random.choice(idx_pos, real_samples, replace=False)
    selected_neg = np.random.choice(idx_neg, real_samples, replace=False)

    # 4. 合并索引
    selected_indices = np.concatenate([selected_pos, selected_neg])

    # 5. [关键] 必须打乱，否则前面全是1后面全是0
    np.random.shuffle(selected_indices)

    # 6. 真正加载数据到内存
    # 只有这一步才会把数据读入 RAM
    X_balanced = X_full[selected_indices].astype(np.float32)
    y_balanced = y_full[selected_indices].astype(np.float32)

    print(f"✅ 平衡完成: 总数 {len(y_balanced)}, 正负比 1:1")
    return X_balanced, y_balanced


def save_best_bands(bands, epsilon, reward):
    """
    保存当前的特征波段配置
    文件名格式：保存时间-Epsilon-Reward.json
    """
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # 格式化文件名：20260122-163000-Eps0.80.json
    filename = f"{time_str}-Eps{epsilon:.4f}.json"
    save_path = os.path.join(AUTO_SAVE_DIR, filename)

    save_data = {
        "description": f"Auto-saved at Epsilon {epsilon:.4f}, Reward {reward:.4f}",
        "timestamp": time_str,
        "epsilon": epsilon,
        "reward": reward,
        "count": len(bands),
        "selected_bands": [int(b) for b in bands]
    }

    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=4)

    print(f"💾 [Auto Save] 已保存波段配置: {filename}")


def train_dqn():
    # 1. 加载全量数据 (Lazy Load)
    X_full, y_full = load_data()
    num_total_bands = X_full.shape[1]

    # 2. [修改] 获取平衡的 DRL 专用数据集
    X_drl, y_drl = prepare_balanced_drl_data(X_full, y_full, SAMPLES_PER_CLASS)

    # 3. 再次划分为 k-NN 的 训练集 (Fit) 和 验证集 (Score)
    # 这里不需要再 stratify，因为已经是 1:1 了，普通 shuffle split 即可
    X_reward_train, X_reward_val, y_reward_train, y_reward_val = train_test_split(
        X_drl, y_drl, test_size=0.4, random_state=42
    )

    print(f"📊 DRL 奖励计算集 (用于 k-NN):")
    print(f"   - Fit Set  : {X_reward_train.shape} (用于构建分类器)")
    print(f"   - Val Set  : {X_reward_val.shape} (用于计算 OA)")

    # 4. 初始化 Agent
    agent = BandSelectionAgent(num_total_bands)
    print(f"\n🔥 开始训练 D3QN-SBS (目标: {NUM_BANDS_TO_SELECT} 波段)...")

    best_reward = -float('inf')
    best_bands = []

    # === 自动保存逻辑初始化 ===
    # 初始阈值设为 0.8 (因为初始 epsilon 是 1.0，第一次下降 0.2 后保存)
    next_save_threshold = 0.8

    for e in range(TOTAL_EPISODES):
        state = np.zeros(num_total_bands)  # 初始状态
        selected_bands = []
        episode_reward = 0

        for step in range(NUM_BANDS_TO_SELECT):
            # 获取动作
            action = agent.get_action(state, selected_bands)

            # 计算奖励 (使用平衡数据集计算 OA)
            reward = calculate_reward_supervised(
                selected_bands, action,
                X_reward_train, y_reward_train,
                X_reward_val, y_reward_val
            )

            # 更新状态
            next_state = state.copy()
            next_state[action] = 1

            done = (len(selected_bands) == NUM_BANDS_TO_SELECT - 1)

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            selected_bands.append(action)
            episode_reward += reward

        agent.update_target_network()

        # Epsilon 衰减
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # === 自动保存检查逻辑 ===
        # 检查当前 Epsilon 是否跌破了下一次保存的阈值
        if agent.epsilon <= next_save_threshold:
            # 只有当找到了有效波段时才保存
            if best_bands:
                save_best_bands(best_bands, agent.epsilon, best_reward)

            # 更新下一个阈值
            if agent.epsilon > 0.2:
                next_save_threshold -= 0.2  # 高探索阶段：每降 0.2 保存
            else:
                next_save_threshold -= 0.01  # 低探索阶段：每降 0.01 保存

            # 防止阈值变成负数（虽然 epsilon_min 是 0.01，但逻辑上保险一点）
            if next_save_threshold < 0:
                next_save_threshold = 0

        # 记录最佳
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_bands = sorted(selected_bands)
            print(f"🌟 [New Best] Ep {e + 1} | Reward: {episode_reward:.4f} | Bands: {best_bands}")

        if (e + 1) % 10 == 0:
            print(f"Episode {e + 1}/{TOTAL_EPISODES} | Reward: {episode_reward:.4f} | Epsilon: {agent.epsilon:.4f}")

    print(f"\n🏆 最终筛选结果: {best_bands}")
    return best_bands


if __name__ == "__main__":
    # 1. 运行 DRL 筛选波段
    final_bands = train_dqn()

    # 2. 重新加载少量数据用于绘图展示
    # 使用之前定义的 load_data 获取 X_full 和 y_full
    X_full, y_full = load_data()
    # 采样一部分 PET 样本进行可视化
    X_plot, y_plot = prepare_balanced_drl_data(X_full, y_full, samples_per_class=1000)

    # 3. [新增功能] 执行验证与绘图
    # 注意：确保 visualization.py 中有这个函数
    try:
        from visualization import visualize_and_verify_pet_bands

        visualize_and_verify_pet_bands(
            X_data=X_plot,
            y_data=y_plot,
            selected_bands=final_bands,
            save_path="pet_feature_validation.png"
        )
    except ImportError:
        print("⚠️ 未找到 visualize_and_verify_pet_bands 函数，跳过绘图")

    # 4. 保存 JSON 配置文件 (最终结果)
    output_filename = "best_bands_config.json"
    save_data = {
        "description": "D3QN-SBS Final Result",
        "count": len(final_bands),
        "selected_bands": [int(b) for b in final_bands]
    }
    with open(output_filename, "w") as f:
        json.dump(save_data, f, indent=4)
    print(f"💾 最终配置文件已保存: {output_filename}")