
import os
# 禁用 oneDNN 优化，解决 Windows 下的死锁问题
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# (可选) 如果你怀疑显卡有问题，取消下面这行的注释来强制使用 CPU 跑跑看
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras import mixed_precision

# ... 其他 imports ...

# ✅ 1. 设置混合精度策略 (利用 3090 Ti 的 Tensor Cores)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# ✅ 2. 显存自增长配置 (你代码里已经有了，保持住)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import time
import glob
import gc

# 引用现有模块
from entropy_utils import precompute_entropies, precompute_mutual_information
from agent import BandSelectionAgent
from reward_utils import calculate_reward
from visualization import visualize_spectral_selection

# ================= 🔧 配置区域 =================
DATA_DIR = r"E:\SPEDATA\NP_data"
CONFIG_OUTPUT_FILE = "best_bands_config.json"
NUM_BANDS_TO_SELECT = 30
TOTAL_EPISODES = 300
SAMPLE_SIZE = 12000
ALPHA = 0.7

# 显存配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


# ================= 🧠 数据加载逻辑 =================
def load_multiclass_data(data_dir, total_samples=10000):
    all_files = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    class_map = {"PET": 1, "CC": 2, "PA": 3}
    file_groups = {0: [], 1: [], 2: [], 3: []}

    for f in all_files:
        path_upper = os.path.dirname(f).upper()
        label = 0
        for key, val in class_map.items():
            if key in path_upper:
                label = val
                break
        file_groups[label].append(f)

    target_per_class = total_samples // 4
    X_list, y_list = [], []

    def sample_category(file_list, label, count):
        if not file_list: return
        collected_x = []
        current_count = 0
        np.random.shuffle(file_list)
        for f in file_list:
            if current_count >= count: break
            try:
                data = np.load(f)
                flat = data.reshape(-1, data.shape[2])
                if label != 0:
                    flat = flat[np.mean(flat, axis=1) > 0.05]
                take = min(len(flat), 600)
                idx = np.random.choice(len(flat), take, replace=False)
                collected_x.append(flat[idx])
                current_count += take
            except:
                pass
        if collected_x:
            X_part = np.concatenate(collected_x, axis=0)[:count]
            X_list.append(X_part)
            y_list.append(np.full(len(X_part), label))

    for lbl in [1, 2, 3, 0]: sample_category(file_groups[lbl], lbl, target_per_class)
    X, y = np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]


# ================= 🏋️‍♂️ 目标特异性训练 =================
def train_for_target(target_name, target_label_id, X_all, y_all):
    print(f"\n🎯 训练目标: [{target_name}]")
    num_bands = X_all.shape[1]
    y_binary = (y_all == target_label_id).astype(int)

    # === 👇 之前缺失的部分 (必须补上) 👇 ===
    print("📊 正在计算信息熵 (Entropy)...")
    entropies = precompute_entropies(X_all)

    print("🔍 正在计算互信息 (Mutual Information)...")
    mi_scores = precompute_mutual_information(X_all, y_binary)
    # ========================================

    # 归一化
    mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-6)
    entropies = (entropies - np.min(entropies)) / (np.max(entropies) - np.min(entropies) + 1e-6)

    # Debug 打印
    print("🤖 正在清理 Session...")
    tf.keras.backend.clear_session()

    print("🏗️ 正在初始化 BandSelectionAgent 模型...")
    agent = BandSelectionAgent(num_bands)
    print("✅ 模型初始化完成！准备开始训练循环...")

    best_reward, best_bands = -np.inf, []

    for e in range(TOTAL_EPISODES):
        state, selected_bands, episode_reward = np.zeros(num_bands), [], 0

        # 打印第一轮进度，确保进入循环
        if e == 0: print("🔄 进入第 1 个 Episode...")

        for t in range(NUM_BANDS_TO_SELECT):
            action = agent.get_action(state, selected_bands)
            reward = calculate_reward(selected_bands, action, entropies, mi_scores, alpha=ALPHA)
            next_state = state.copy()
            next_state[action] = 1
            agent.remember(state, action, reward, next_state, (t == NUM_BANDS_TO_SELECT - 1))
            agent.train()
            state, selected_bands, episode_reward = next_state, selected_bands + [action], episode_reward + reward

        if agent.epsilon > agent.epsilon_min: agent.epsilon *= agent.epsilon_decay
        if episode_reward > best_reward:
            best_reward, best_bands = episode_reward, sorted(selected_bands)
        if (e + 1) % 50 == 0: print(f" Episode {e + 1}/{TOTAL_EPISODES} | Best Reward: {best_reward:.4f}")

    visualize_spectral_selection(X_all, y_all, best_bands, save_path=f"analysis_{target_name}.png")
    return [int(b) for b in best_bands]

if __name__ == "__main__":
    X_global, y_global = load_multiclass_data(DATA_DIR, SAMPLE_SIZE)

    # ================= 🛡️ 新增修复代码 =================
    print(f"🧹 清洗前数据范围: Min={np.min(X_global):.2f}, Max={np.max(X_global)}")

    if not np.isfinite(X_global).all():
        print("⚠️ 警告：数据中仍包含非有限值，正在强制裁剪...")
        X_global = np.clip(X_global, 0, 100)  # 强制裁剪到 0-100 防止溢出

    print("✅ 数据清洗完成。")

    results = {}
    for name, lid in [("PET", 1), ("CC", 2), ("PA", 3)]:
        results[name] = train_for_target(name, lid, X_global, y_global)
        gc.collect()

    final_config = {
        "targets": results,
        "all_unique_bands": sorted(list(set(sum(results.values(), []))))
    }
    with open(CONFIG_OUTPUT_FILE, 'w') as f:
        json.dump(final_config, f, indent=4)
    print(f"✅ 配置文件已保存: {CONFIG_OUTPUT_FILE}")