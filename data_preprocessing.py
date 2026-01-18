import numpy as np
import os
import json
import tensorflow as tf
import time
import glob
import gc

# 引用现有模块
from entropy_utils import precompute_entropies, precompute_mutual_information
from agent import BandSelectionAgent
from reward_utils import calculate_reward

# 如果你没有 visualization 模块，可以注释掉下面这行
try:
    from visualization import visualize_spectral_selection
except ImportError:
    visualize_spectral_selection = None

# ================= 🔧 配置区域 =================
# 数据路径 (指向 save_data.py 输出的文件夹)
DATA_DIR = r"E:\SPEDATA\NP_data"  # 请确保路径正确

# 结果保存配置
CONFIG_OUTPUT_FILE = "best_bands_config.json"
MODEL_CHECKPOINT_DIR = "checkpoints"

# DRL 超参数
NUM_BANDS_TO_SELECT = 30  # 每种材质选多少个波段
TOTAL_EPISODES = 300  # 每种材质训练多少轮 (建议 300-500)
SAMPLE_SIZE = 12000  # 采样总点数
ALPHA = 0.7  # 奖励权重

# 显存配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU 显存按需分配已启用")
    except RuntimeError as e:
        print(e)


# ================= 🧠 增强版数据加载 =================

def load_multiclass_data(data_dir, total_samples=10000):
    """
    加载数据并返回原始多类别标签。
    0: Background/Other
    1: PET
    2: CC
    3: PA
    """
    print(f"🚀 [IO] 正在从 {data_dir} 加载多材质数据...")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ 找不到数据目录: {data_dir}")

    all_files = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)

    # 定义类别映射
    # 确保 save_data.py 生成的文件夹名称包含这些关键字
    class_map = {
        "PET": 1,
        "CC": 2,
        "PA": 3
    }

    file_groups = {0: [], 1: [], 2: [], 3: []}

    for f in all_files:
        path_upper = os.path.dirname(f).upper()
        label = 0  # 默认为背景/其他
        for key, val in class_map.items():
            if key in path_upper:
                label = val
                break
        file_groups[label].append(f)

    print(
        f"   📊 文件分布: PET={len(file_groups[1])}, CC={len(file_groups[2])}, PA={len(file_groups[3])}, Other={len(file_groups[0])}")

    # 每类采样数量 (尽量平衡)
    target_per_class = total_samples // 4

    X_list = []
    y_list = []

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

                # 简单的背景过滤 (去除全黑像素)
                if label != 0:
                    intensity = np.mean(flat, axis=1)
                    flat = flat[intensity > 0.05]

                if len(flat) == 0: continue

                # 随机采样
                take = min(len(flat), 600)
                idx = np.random.choice(len(flat), take, replace=False)
                collected_x.append(flat[idx])
                current_count += take
            except:
                pass

        if collected_x:
            X_part = np.concatenate(collected_x, axis=0)
            if len(X_part) > count: X_part = X_part[:count]
            y_part = np.full(len(X_part), label)
            X_list.append(X_part)
            y_list.append(y_part)

    # 执行采样
    for lbl in [1, 2, 3, 0]:
        sample_category(file_groups[lbl], lbl, target_per_class)

    if not X_list:
        raise ValueError("❌ 未加载到数据！请检查路径或 save_data.py 是否正确运行。")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    print(f"✅ 数据加载完毕: Shape {X.shape}, Labels {np.unique(y)}")
    return X[idx], y[idx]


# ================= 🏋️‍♂️ 独立训练流程 =================

def train_for_target(target_name, target_label_id, X_all, y_all):
    """
    针对特定材质进行 DRL 训练
    :param target_name: 'PET', 'CC', or 'PA'
    :param target_label_id: 1, 2, or 3
    :param X_all: 所有光谱数据
    :param y_all: 原始标签 (0,1,2,3)
    """
    print(f"\n" + "=" * 60)
    print(f"🎯 开始训练目标: [{target_name}] (Label {target_label_id} vs Rest)")
    print("=" * 60)

    num_bands = X_all.shape[1]

    # 1. 构造 One-vs-Rest 二分类标签
    # 目标材质 = 1, 其他所有材质(包括背景) = 0
    y_binary = (y_all == target_label_id).astype(int)

    pos_samples = np.sum(y_binary == 1)
    neg_samples = np.sum(y_binary == 0)
    print(f"   样本分布 -> 正样本({target_name}): {pos_samples} | 负样本(Rest): {neg_samples}")

    if pos_samples < 100:
        print(f"⚠️ 警告: {target_name} 样本太少，训练效果可能不佳！")

    # 2. 计算针对该目标的互信息 (关键步骤!)
    # 这会告诉 Agent 哪些波段最能区分 [目标] 和 [其他]
    print(f"⏳ 正在计算 {target_name} 的专属互信息...")
    entropies = precompute_entropies(X_all)  # 熵是通用的
    mi_scores = precompute_mutual_information(X_all, y_binary)  # 互信息是特异的

    # 归一化
    if np.max(entropies) != np.min(entropies):
        entropies = (entropies - np.min(entropies)) / (np.max(entropies) - np.min(entropies))
    if np.max(mi_scores) != np.min(mi_scores):
        mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores))

    # 3. 初始化新的 Agent
    tf.keras.backend.clear_session()  # 清理旧图
    agent = BandSelectionAgent(num_bands)

    best_reward = -np.inf
    best_bands = []

    # 4. 训练循环
    start_time = time.time()
    for e in range(TOTAL_EPISODES):
        state = np.zeros(num_bands)
        selected_bands = []
        episode_reward = 0

        for t in range(NUM_BANDS_TO_SELECT):
            action = agent.get_action(state, selected_bands)

            # 使用针对该目标的 MI 计算奖励
            reward = calculate_reward(selected_bands, action, entropies, mi_scores, alpha=ALPHA)

            next_state = state.copy()
            next_state[action] = 1
            done = (len(selected_bands) == NUM_BANDS_TO_SELECT - 1)

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            selected_bands.append(action)
            episode_reward += reward

            if done:
                agent.update_target_network()
                break

        # 衰减 & 记录
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_bands = sorted(selected_bands)

        if (e + 1) % 50 == 0:
            print(f"   Episode {e + 1}/{TOTAL_EPISODES} | Reward: {episode_reward:.4f} | Best: {best_reward:.4f}")

    print(f"✅ {target_name} 训练完成! 耗时: {(time.time() - start_time):.1f}s")
    print(f"💎 选出的特征波段: {best_bands}")

    # 5. 生成该材质的可视化图 (可选)
    if visualize_spectral_selection:
        try:
            visualize_spectral_selection(
                X_all, y_all, best_bands,
                save_path=f"analysis_{target_name}.png"
            )
        except Exception as e:
            print(f"可视化跳过: {e}")

    return [int(b) for b in best_bands]


# ================= 🚀 主程序 =================

if __name__ == "__main__":
    if not os.path.exists(MODEL_CHECKPOINT_DIR):
        os.makedirs(MODEL_CHECKPOINT_DIR)

    # 1. 一次性加载所有数据
    X_global, y_global = load_multiclass_data(DATA_DIR, SAMPLE_SIZE)

    results = {}

    # 2. 依次对 PET(1), CC(2), PA(3) 进行训练
    targets = [
        ("PET", 1),
        ("CC", 2),
        ("PA", 3)
    ]

    for name, label_id in targets:
        # 检查数据中是否存在该标签
        if np.sum(y_global == label_id) == 0:
            print(f"❌ 跳过 {name}: 数据集中没有 Label {label_id} 的样本！")
            results[f"{name}_bands"] = []
            continue

        # 执行训练
        selected = train_for_target(name, label_id, X_global, y_global)
        results[f"{name}_bands"] = selected

        # 显存清理
        gc.collect()

    # 3. 保存最终汇总结果
    print("\n" + "=" * 60)
    print("💾 正在保存多材质波段配置...")

    # 结构化输出
    final_config = {
        "description": "Multi-material characteristic bands selected by DRL",
        "targets": {
            "PET": results.get("PET_bands", []),
            "CC": results.get("CC_bands", []),
            "PA": results.get("PA_bands", [])
        },
        # 为了兼容旧代码，我们可以把所有波段合并去重作为 selected_bands
        # 或者你可以修改后续代码来读取 specific bands
        "all_unique_bands": sorted(list(set(
            results.get("PET_bands", []) +
            results.get("CC_bands", []) +
            results.get("PA_bands", [])
        )))
    }

    with open(CONFIG_OUTPUT_FILE, 'w') as f:
        json.dump(final_config, f, indent=4)

    print(f"✅ 配置文件已保存: {os.path.abspath(CONFIG_OUTPUT_FILE)}")
    print(f"   包含 PET 波段: {len(final_config['targets']['PET'])} 个")
    print(f"   包含 CC  波段: {len(final_config['targets']['CC'])} 个")
    print(f"   包含 PA  波段: {len(final_config['targets']['PA'])} 个")
    print("=" * 60)