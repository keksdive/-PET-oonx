import numpy as np
import os
import json
import tensorflow as tf
import time

# 引用现有模块
from entropy_utils import precompute_entropies, precompute_mutual_information
from agent import BandSelectionAgent
from reward_utils import calculate_reward

# ================= 🔧 配置区域 =================
# [关键] 指向 save_data.py 输出的文件夹
DATA_DIR = r"I:\SPEDATA\NP_data"

# 结果保存配置
CONFIG_OUTPUT_FILE = "best_bands_config.json"
MODEL_CHECKPOINT_DIR = "checkpoints"

# DRL 超参数
NUM_BANDS_TO_SELECT = 30  # 最终选择多少个波段
TOTAL_EPISODES = 500  # 训练轮数
SAMPLE_SIZE = 10000  # 采样点数
ALPHA = 0.7  # 奖励权重

# 显存配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU 显存按需分配已启用")
    except RuntimeError as e:
        print(e)


# ================= 🧠 数据加载逻辑 =================

# 将此函数替换原来的 load_representative_data_for_drl

def load_representative_data_for_drl():
    print("🚀 [DRL] 正在加载混合样本 (PET + CC + PA)...")

    # === 修改 1: 扩展数据集配置，增加 use_json 标记 ===
    dataset_configs = [
        # 1. PET 文件夹 (需要 JSON 区分瓶片和杂质)
        {
            "root": r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET",
            "json_subdir": "fake_images",
            "is_pet_folder": True,
            "use_json": True  # 使用 JSON 解析
        },
        # 2. CC (纯材质，无需 JSON，整张图除了背景都是 CC)
        {
            "root": r"I:\SPEDATA\高谱相机数据集\训练集\no_PET\CC",
            "json_subdir": None,
            "is_pet_folder": False,
            "use_json": False  # 不用 JSON，自动提取
        },
        # 3. PA (纯材质，无需 JSON)
        {
            "root": r"I:\SPEDATA\高谱相机数据集\训练集\no_PET\PA",
            "json_subdir": None,
            "is_pet_folder": False,
            "use_json": False
        }
    ]

    white = load_calibration_file(WHITE_REF_HDR)
    dark = load_calibration_file(DARK_REF_HDR)
    denom = (white - dark)
    denom[denom == 0] = 1e-6

    collected_X = []
    collected_y = []

    count_pet = 0
    count_hard_neg = 0  # PP, CC, PA
    count_soft_neg = 0  # 背景

    # 目标采样数
    TARGET_PER_CLASS = 3000

    for config in dataset_configs:
        root_dir = config["root"]
        is_pet_source = config["is_pet_folder"]
        use_json = config["use_json"]  # 获取标记

        if not os.path.exists(root_dir):
            print(f"⚠️ 路径不存在跳过: {root_dir}")
            continue

        files = [f for f in os.listdir(root_dir) if f.endswith('.spe')]

        for fname in files:
            # 提前停止
            if is_pet_source and count_pet >= TARGET_PER_CLASS and count_soft_neg >= TARGET_PER_CLASS: continue
            if not is_pet_source and count_hard_neg >= TARGET_PER_CLASS: continue

            try:
                # 加载并校准
                hdr_path = os.path.join(root_dir, fname.replace('.spe', '.hdr'))
                if not os.path.exists(hdr_path): continue

                raw = np.array(envi.open(hdr_path, os.path.join(root_dir, fname)).load(), dtype=np.float32)

                # 维度修正
                if raw.shape[1] == 208 and raw.shape[2] != 208:
                    raw = np.transpose(raw, (0, 2, 1))

                # 波段对齐 (处理可能的 206/208 问题)
                if raw.shape[2] != denom.shape[2]:
                    # 简单裁剪或报错，这里假设已经对齐或使用之前 save_data 的逻辑
                    # 为简单起见，这里假设维度一致，如果不一致建议先用 save_data 处理成 npy
                    pass

                calib = (raw - dark) / denom

                mask = None

                # === 修改 2: 分情况处理 Mask ===
                if use_json:
                    # 原有逻辑：读取 JSON
                    json_subdir = config["json_subdir"]
                    base_name = os.path.splitext(fname)[0]
                    json_path = os.path.join(root_dir, json_subdir, base_name + ".json")
                    if not os.path.exists(json_path):
                        json_path = os.path.join(root_dir, base_name + ".json")

                    if os.path.exists(json_path):
                        mask = get_mask_from_json(json_path, (calib.shape[0], calib.shape[1]))
                else:
                    # === 新增逻辑：自动背景去除 ===
                    # 计算平均亮度
                    intensity = np.mean(calib, axis=2)
                    # 阈值 0.05 (根据数据调整，背景通常接近 0)
                    fg_mask = (intensity > 0.05)

                    if np.sum(fg_mask) > 100:  # 只要有足够的前景
                        mask = np.zeros((calib.shape[0], calib.shape[1]), dtype=np.uint8)
                        # 标记为 2 (强负样本)
                        mask[fg_mask] = 2

                if mask is None: continue

                # === 后续提取逻辑保持不变 ===
                flat_data = calib.reshape(-1, calib.shape[2])
                flat_mask = mask.reshape(-1)

                idx_pet = np.where(flat_mask == 1)[0]
                idx_mat = np.where(flat_mask == 2)[0]  # CC, PA, PP 都在这里
                idx_bg = np.where(flat_mask == 0)[0]

                # PET
                if is_pet_source and len(idx_pet) > 0 and count_pet < TARGET_PER_CLASS:
                    take = min(len(idx_pet), 200)
                    sel = np.random.choice(idx_pet, take, replace=False)
                    collected_X.append(flat_data[sel])
                    collected_y.append(np.ones(take))  # y=1
                    count_pet += take

                # 强负样本 (CC, PA, PP) -> y=0
                if len(idx_mat) > 0 and count_hard_neg < TARGET_PER_CLASS:
                    take = min(len(idx_mat), 200)
                    sel = np.random.choice(idx_mat, take, replace=False)
                    collected_X.append(flat_data[sel])
                    collected_y.append(np.zeros(take))  # y=0
                    count_hard_neg += take

                # 背景 -> y=0
                if is_pet_source and len(idx_bg) > 0 and count_soft_neg < TARGET_PER_CLASS:
                    take = min(len(idx_bg), 100)
                    sel = np.random.choice(idx_bg, take, replace=False)
                    collected_X.append(flat_data[sel])
                    collected_y.append(np.zeros(take))  # y=0
                    count_soft_neg += take

            except Exception as e:
                # print(f"Skipping {fname}: {e}")
                pass

    if not collected_X: raise ValueError("没有加载到数据，请检查路径和校准文件！")

    X = np.concatenate(collected_X, axis=0)
    y = np.concatenate(collected_y, axis=0)

    print(f"✅ DRL 采样完成: PET={np.sum(y == 1)}, Non-PET(CC+PA+BG)={np.sum(y == 0)}")
    return X, y

# ================= 🚀 主训练流程 =================


import glob


def load_data_from_npy(data_dir, total_samples=10000):
    """
    从 save_data.py 生成的 .npy 文件中加载数据
    PET 文件夹 -> 标签 1
    其他文件夹 (CC, PA, PP, OTHER) -> 标签 0
    """
    print(f"🚀 [DRL] 正在从 {data_dir} 加载 .npy 数据...")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ 找不到数据目录: {data_dir}")

    all_files = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    if not all_files:
        raise ValueError("❌ 目录下没有找到 .npy 文件，请先运行 save_data.py")

    X_list = []
    y_list = []

    # 简单的平衡采样逻辑
    pet_files = [f for f in all_files if "PET" in os.path.dirname(f)]
    other_files = [f for f in all_files if "PET" not in os.path.dirname(f)]

    print(f"   发现 PET 文件: {len(pet_files)} 个, 非 PET 文件: {len(other_files)} 个")

    # 每个类别采样的目标数量
    target_per_class = total_samples // 2

    def sample_from_files(file_list, target_count, label):
        current_count = 0
        collected_x = []

        # 随机打乱文件顺序
        np.random.shuffle(file_list)

        for f in file_list:
            if current_count >= target_count: break
            try:
                data = np.load(f)  # shape (H, W, Bands)
                # 展平
                flat = data.reshape(-1, data.shape[2])

                # 从每张图中随机取 500 个点 (避免单张图主导)
                n_take = min(len(flat), 500)
                idx = np.random.choice(len(flat), n_take, replace=False)

                collected_x.append(flat[idx])
                current_count += n_take
            except Exception as e:
                print(f"⚠️ 读取错误 {f}: {e}")

        if not collected_x: return np.array([]), np.array([])

        X_part = np.concatenate(collected_x, axis=0)
        y_part = np.full(len(X_part), label)

        # 如果采多了，截断
        if len(X_part) > target_count:
            idx = np.random.choice(len(X_part), target_count, replace=False)
            X_part = X_part[idx]
            y_part = y_part[idx]

        return X_part, y_part

    # 采样 PET (Label 1)
    X_pos, y_pos = sample_from_files(pet_files, target_per_class, 1)

    # 采样 非PET (Label 0)
    X_neg, y_neg = sample_from_files(other_files, target_per_class, 0)

    if len(X_pos) == 0 or len(X_neg) == 0:
        raise ValueError("❌ 采样失败：某一类数据为空，请检查 .npy 文件路径结构")

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    # 再次打乱
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    print(f"✅ 数据加载完成: 总数 {len(X)}, 正样本 {np.sum(y == 1)}, 负样本 {np.sum(y == 0)}")
    return X[idx], y[idx]








if __name__ == "__main__":
    if not os.path.exists(MODEL_CHECKPOINT_DIR):
        os.makedirs(MODEL_CHECKPOINT_DIR)

    # 1. 加载数据
    X_sample, y_sample = load_data_from_npy(DATA_DIR, SAMPLE_SIZE)

    num_bands = X_sample.shape[1]
    print(f"🔍 波段总数: {num_bands}")

    # 2. 预计算熵和互信息
    print("⏳ 正在预计算互信息 (这决定了波段的判别力)...")
    entropies = precompute_entropies(X_sample)

    # 互信息算法会自动寻找能区分所有类别的波段
    mi_matrix = precompute_mutual_information(X_sample, y_sample)

    # 归一化
    if np.max(entropies) != np.min(entropies):
        entropies = (entropies - np.min(entropies)) / (np.max(entropies) - np.min(entropies))
    if np.max(mi_matrix) != np.min(mi_matrix):
        mi_matrix = (mi_matrix - np.min(mi_matrix)) / (np.max(mi_matrix) - np.min(mi_matrix))

    print("✅ 互信息计算完毕。")

    # 3. 初始化 DRL Agent
    agent = BandSelectionAgent(num_bands)

    best_reward = -np.inf
    best_bands = []

    print(f"🚀 开始训练 DRL Agent ({TOTAL_EPISODES} Episodes)...")
    start_time = time.time()

    for e in range(TOTAL_EPISODES):
        state = np.zeros(num_bands)
        selected_bands = []
        episode_reward = 0

        for t in range(NUM_BANDS_TO_SELECT):
            # --- [修复核心] ---
            # 原代码: action = agent.act(state, available_bands=range(num_bands))
            # 修正为: 使用 get_action 并传入 selected_bands 以便屏蔽已选波段
            action = agent.get_action(state, selected_bands)
            # ------------------

            # 奖励计算
            reward = calculate_reward(selected_bands, action, entropies, mi_matrix, alpha=ALPHA)

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

        # 探索率衰减
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # 记录最佳
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_bands = sorted(selected_bands)

        if (e + 1) % 10 == 0:
            print(
                f"Episode {e + 1}/{TOTAL_EPISODES} | Reward: {episode_reward:.4f} | Epsilon: {agent.epsilon:.2f} | Best Bands: {len(best_bands)}")

    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"🏆 训练结束 (耗时 {total_time / 60:.1f} min)")
    print(f"💎 最佳波段组合 (Reward: {best_reward:.4f}):")
    print(best_bands)
    print("=" * 50)

    # 4. 保存结果
    output_data = {
        "selected_bands": [int(b) for b in best_bands],
        "reward": float(best_reward)
    }

    with open(CONFIG_OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"💾 波段配置已保存至: {CONFIG_OUTPUT_FILE}")