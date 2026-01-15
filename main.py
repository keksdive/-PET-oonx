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


def load_representative_data_for_drl():
    print("🚀 正在加载 DRL 训练数据...")
    white = load_calibration_file(WHITE_REF_HDR)
    dark = load_calibration_file(DARK_REF_HDR)
    denom = (white - dark)
    denom[denom == 0] = 1e-6

    collected_X = []
    collected_y = []

    pet_count = 0
    non_pet_count = 0

    for root, dirs, files in os.walk(TRAIN_DATA_ROOT):
        for fname in files:
            if not fname.endswith('.spe'): continue

            json_path = os.path.join(root, fname.replace('.spe', '.json'))
            if not os.path.exists(json_path): continue

            try:
                # 快速检查 JSON，如果该图没有我们要的标签，直接跳过加载图像（省时间）
                with open(json_path, 'r', encoding='utf-8') as f:
                    jdata = json.load(f)
                    has_pet = any(
                        'pet' in s['label'].lower() and 'no_pet' not in s['label'].lower() for s in jdata['shapes'])
                    # 如果当前 PET 样本严重不足，优先加载含 PET 的图
                    if pet_count < SAMPLE_SIZE // 2 and not has_pet:
                        continue

                # 加载图像
                hdr_path = os.path.join(root, fname + ".hdr")
                if not os.path.exists(hdr_path): hdr_path = os.path.splitext(os.path.join(root, fname))[0] + ".hdr"

                # 修复 Header
                if os.path.exists(hdr_path):
                    with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
                        if 'byte order' not in f.read().lower():
                            with open(hdr_path, 'a') as fa: fa.write('\nbyte order = 0')

                raw = envi.open(hdr_path, os.path.join(root, fname)).load()
                if raw.shape[1] == 208: raw = np.transpose(raw, (0, 2, 1))
                calib = (raw.astype(np.float32) - dark) / denom

                mask = get_mask_from_json(json_path, (calib.shape[0], calib.shape[1]))

                # 采样 PET (Label 1)
                idx1 = np.where(mask == 1)
                n_p = len(idx1[0])
                if n_p > 0:
                    # 动态调整采样量：如果 PET 缺口大，就多采点
                    needed = (SAMPLE_SIZE // 2) - pet_count
                    take = min(n_p, max(100, needed // 5))  # 每次最少采100，除非不够
                    indices = np.random.choice(n_p, size=take, replace=False)
                    collected_X.append(calib[idx1[0][indices], idx1[1][indices], :])
                    collected_y.append(np.ones(take))
                    pet_count += take

                # 采样 NO_PET (Label 2)
                idx2 = np.where(mask == 2)
                n_np = len(idx2[0])
                if n_np > 0 and non_pet_count < SAMPLE_SIZE // 2:
                    take = min(n_np, 100)
                    indices = np.random.choice(n_np, size=take, replace=False)
                    collected_X.append(calib[idx2[0][indices], idx2[1][indices], :])
                    collected_y.append(np.zeros(take))
                    non_pet_count += take

                print(f"  -> 进度: PET {pet_count} | Non-PET {non_pet_count} | 当前文件: {fname}", end='\r')

                if pet_count >= SAMPLE_SIZE // 2 and non_pet_count >= SAMPLE_SIZE // 2:
                    break

            except Exception as e:
                print(f"\nSkip {fname}: {e}")

        if pet_count >= SAMPLE_SIZE // 2 and non_pet_count >= SAMPLE_SIZE // 2:
            break

    print("\n")
    if not collected_X:
        raise ValueError("❌ 未找到任何有效数据！请检查路径和 JSON 标签。")

    X = np.concatenate(collected_X, axis=0)
    y = np.concatenate(collected_y, axis=0)

    print(f"✅ DRL 数据加载统计: 总数 {len(y)}, PET(1): {np.sum(y == 1)}, 背景(0): {np.sum(y == 0)}")

    if np.sum(y == 1) == 0:
        raise ValueError("⛔【致命错误】未检测到任何 PET 样本！标签分布全是 0。\n"
                         "请检查：1. JSON文件中 PET 的标签是否包含 'pet' 且不含 'no_pet'？\n"
                         "2. 是否所有图片的 JSON 都只有背景标注？")

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