import numpy as np
import os
import random
import spectral.io.envi as envi
import cv2
import json
import tensorflow as tf

# 引用你的模块
from entropy_utils import precompute_entropies, precompute_mutual_information
from agent import BandSelectionAgent
from reward_utils import calculate_reward

# 引入数据预处理中的 SNV (如果你的 data_preprocessing.py 里没有 apply_snv，请先添加，或者使用下面的内置函数)

# ================= 🔧 配置区域 =================
TRAIN_DATA_ROOT = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）"
WHITE_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\white_ref.hdr"
DARK_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\black_ref.hdr"

NUM_BANDS_TO_SELECT = 30
TOTAL_EPISODES = 300
ALPHA = 0.7
SAMPLE_SIZE = 5000

# 显存配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU 显存按需分配已开启")
    except RuntimeError as e:
        print(e)


# ===============================================

def apply_snv(spectra):
    """【论文优化】标准正态变量变换 (SNV)"""
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    std[std == 0] = 1e-6
    return (spectra - mean) / std


def load_calibration_file(hdr_path):
    base = os.path.splitext(hdr_path)[0]
    spe = base + ".spe"
    if not os.path.exists(spe) and os.path.exists(base): spe = base
    img = envi.open(hdr_path, spe).load()
    if img.shape[1] == 208 and img.shape[2] != 208:
        img = np.transpose(img, (0, 2, 1))
    return np.array(img, dtype=np.float32)


def get_mask_from_json(json_path, img_shape):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mask = np.zeros(img_shape, dtype=np.uint8)
    labels_found = []
    for shape in data['shapes']:
        lbl = shape['label'].lower()
        labels_found.append(lbl)
        pts = np.array(shape['points'], dtype=np.int32)
        if 'no_pet' in lbl:
            cv2.fillPoly(mask, [pts], 2)  # NO_PET
        elif 'pet' in lbl:
            cv2.fillPoly(mask, [pts], 1)  # PET

    # 调试信息：如果没找到 Mask，打印一下 JSON 里的标签
    if np.sum(mask) == 0:
        # print(f"  ⚠️ Warning: {os.path.basename(json_path)} 中未匹配到 'pet'/'no_pet'。包含标签: {list(set(labels_found))}")
        pass
    return mask


# 修改 main.py

def load_representative_data_for_drl():
    print("🚀 [DRL] 正在加载混合样本 (PET + 强负样本PP)...")

    # 定义两个数据源（和 save_data.py 类似）
    dataset_configs = [
        # 1. PET 文件夹
        {
            "root": r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET",
            "json_subdir": "fake_images",
            "is_pet_folder": True
        },
        # 2. Non-PET 文件夹 (PP, CC等)
        {
            "root": r"I:\Hyperspectral Camera Dataset\Train_Data\no_PET\no_PET(CC醋酸纤维素)",
            "json_subdir": "fake_images",
            "is_pet_folder": False
        }
    ]

    white = load_calibration_file(WHITE_REF_HDR)
    dark = load_calibration_file(DARK_REF_HDR)
    denom = (white - dark)
    denom[denom == 0] = 1e-6

    collected_X = []
    collected_y = []

    # 计数器
    count_pet = 0
    count_hard_neg = 0  # PP, CC
    count_soft_neg = 0  # 背景

    # 目标采样数 (根据你的显存调整，建议 5000-10000)
    TARGET_PER_CLASS = 3000

    for config in dataset_configs:
        root_dir = config["root"]
        json_subdir = config["json_subdir"]
        is_pet_source = config["is_pet_folder"]

        if not os.path.exists(root_dir): continue

        files = [f for f in os.listdir(root_dir) if f.endswith('.spe')]

        for fname in files:
            # 提前停止条件
            if is_pet_source and count_pet >= TARGET_PER_CLASS and count_soft_neg >= TARGET_PER_CLASS: continue
            if not is_pet_source and count_hard_neg >= TARGET_PER_CLASS: continue

            try:
                # ... (加载 hdr, spe, 校准 代码省略，与之前一致) ...
                # 假设得到了 calib 数据

                # 获取 JSON 路径
                base_name = os.path.splitext(fname)[0]
                json_path = os.path.join(root_dir, json_subdir, base_name + ".json")
                if not os.path.exists(json_path):
                    json_path = os.path.join(root_dir, base_name + ".json")

                # 解析 Mask
                mask = get_mask_from_json(json_path, (calib.shape[0], calib.shape[1]))
                if mask is None: continue  # 或者是自动生成的 Mask

                # 提取数据
                flat_data = calib.reshape(-1, calib.shape[2])
                flat_mask = mask.reshape(-1)

                # --- 核心逻辑：区分三类 ---
                # 1. PET (标签 1)
                idx_pet = np.where(flat_mask == 1)[0]
                # 2. 强负样本 (标签 2: PP/CC)
                idx_mat = np.where(flat_mask == 2)[0]
                # 3. 弱负样本 (标签 0: 背景)
                idx_bg = np.where(flat_mask == 0)[0]

                # 采样并添加
                # PET -> y=1
                if is_pet_source and len(idx_pet) > 0 and count_pet < TARGET_PER_CLASS:
                    take = min(len(idx_pet), 200)
                    sel = np.random.choice(idx_pet, take, replace=False)
                    collected_X.append(flat_data[sel])
                    collected_y.append(np.ones(take))  # y=1
                    count_pet += take

                # PP/CC -> y=0 (关键！告诉 DRL 这些不是 PET)
                if len(idx_mat) > 0 and count_hard_neg < TARGET_PER_CLASS:
                    take = min(len(idx_mat), 200)
                    sel = np.random.choice(idx_mat, take, replace=False)
                    collected_X.append(flat_data[sel])
                    collected_y.append(np.zeros(take))  # y=0
                    count_hard_neg += take

                # 背景 -> y=0 (只需要少量，告诉 DRL 区分背景)
                if is_pet_source and len(idx_bg) > 0 and count_soft_neg < TARGET_PER_CLASS:
                    take = min(len(idx_bg), 100)  # 背景少采点，很容易区分
                    sel = np.random.choice(idx_bg, take, replace=False)
                    collected_X.append(flat_data[sel])
                    collected_y.append(np.zeros(take))  # y=0
                    count_soft_neg += take

            except:
                pass

    if not collected_X: raise ValueError("没有加载到数据")

    X = np.concatenate(collected_X, axis=0)
    y = np.concatenate(collected_y, axis=0)

    print(f"✅ DRL 采样完成: PET={np.sum(y == 1)}, Non-PET(PP+BG)={np.sum(y == 0)}")
    return X, y

def train_dqn():
    # 1. 准备数据
    X_full, y_full = load_representative_data_for_drl()

    # 【论文优化】应用 SNV 预处理
    print("🧪 正在应用 SNV 预处理 (Paper Optimization)...")
    X_full = apply_snv(X_full)

    num_total_bands = X_full.shape[1]

    # 2. 计算指标
    print("📊 正在计算 Information Entropy (熵)...")
    entropies = precompute_entropies(X_full)

    print("⚖️ 正在计算 Mutual Information (互信息)...")
    mi_scores = precompute_mutual_information(X_full, y_full)

    # 归一化
    entropies = (entropies - np.min(entropies)) / (np.max(entropies) - np.min(entropies) + 1e-6)
    mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-6)

    # 3. 训练 Agent
    agent = BandSelectionAgent(num_total_bands)

    print(f"\n🔥 开始训练 DRL Agent (Alpha={ALPHA})...")

    # ================= 关键修复：初始化变量 =================
    best_reward = -float('inf')
    best_bands = []  # 👈 之前报错就是因为少了这一行！
    # ======================================================

    for e in range(TOTAL_EPISODES):
        state = np.zeros(num_total_bands)
        selected_bands = []
        episode_reward = 0

        for step in range(NUM_BANDS_TO_SELECT):
            action = agent.get_action(state, selected_bands)
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

        # 更新最佳结果
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_bands = sorted(selected_bands)

        if (e + 1) % 10 == 0:
            print(
                f"Episode {e + 1}/{TOTAL_EPISODES} | Reward: {episode_reward:.4f} | Epsilon: {agent.epsilon:.2f} | Best: {len(best_bands)} bands")

    print("\n" + "=" * 50)
    print(f"🏆 最终推荐的 {len(best_bands)} 个波段 (索引):")
    print(best_bands)
    print("=" * 50)

    return best_bands


if __name__ == "__main__":
    final_bands = train_dqn()

    if not final_bands:
        print("⚠️ 警告：训练未返回波段，使用默认值。")
        final_bands = list(range(30))

    # 保存结果给 pipeline 使用
    config_path = "best_bands_config.json"
    save_data = {"selected_bands": [int(b) for b in final_bands]}

    with open(config_path, "w") as f:
        json.dump(save_data, f)

    print(f"💾 [Auto] 配置已保存至 {config_path}")