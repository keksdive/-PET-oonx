import numpy as np
import os
import random
import spectral.io.envi as envi
import cv2
import json
import gc
import tensorflow as tf

# ================= 🚀 1. 核心依赖检查 =================
try:
    from entropy_utils import precompute_entropies, precompute_mutual_information
    from agent import BandSelectionAgent
    from reward_utils import calculate_reward
except ImportError as e:
    print(f"❌ 缺少依赖文件: {e}")
    print("请确保 agent.py, entropy_utils.py, reward_utils.py 都在同一目录下。")

# ================= 🚀 2. 路径与参数设置 =================
# 光谱数据路径
SPE_ROOT = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET"
# 标注文件路径 (在子目录中)
JSON_ROOT = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET\fake_images"
# 黑白校准文件
WHITE_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\white_ref.hdr"
DARK_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\black_ref.hdr"

# 训练参数
NUM_BANDS_TO_SELECT = 30  # 最终选出多少个波段
TOTAL_EPISODES = 300  # 训练轮数
SAMPLE_PIXELS_PER_IMAGE = 200  # 每张图提取多少个像素点 (防内存溢出)
MAX_TOTAL_SAMPLES = 15000  # 总共用于训练的像素点上限
ALPHA = 0.7  # 奖励函数权重


# =======================================================

def fix_header_byte_order(hdr_path):
    """自动修复 ENVI 头文件"""
    if not os.path.exists(hdr_path): return
    try:
        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        if not any('byte order' in line.lower() for line in lines):
            with open(hdr_path, 'a') as f: f.write('\nbyte order = 0')
    except:
        pass


def load_calib_hdr(hdr_path):
    """加载校准文件"""
    fix_header_byte_order(hdr_path)
    # 自动推断对应的 .spe 文件路径
    spe_path = hdr_path.replace('.hdr', '.spe')
    if not os.path.exists(spe_path):
        spe_path = os.path.splitext(hdr_path)[0] + ".spe"

    img = envi.open(hdr_path, spe_path).load()
    # 统一格式为 (H, W, Bands)
    if img.shape[1] == 208:
        img = np.transpose(img, (0, 2, 1))
    return np.array(img, dtype=np.float32)


def get_mask_from_json(json_path, img_shape):
    """从 JSON 解析标签"""
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mask = np.zeros(img_shape, dtype=np.uint8)
        found = False
        for shape in data['shapes']:
            lbl = shape['label'].lower()
            pts = np.array(shape['points'], dtype=np.int32)
            # 兼容不同标签名
            if 'no_pet' in lbl or 'background' in lbl:
                cv2.fillPoly(mask, [pts], 2)  # 负样本
                found = True
            elif 'pet' in lbl:
                cv2.fillPoly(mask, [pts], 1)  # 正样本
                found = True
        return mask if found else None
    except:
        return None


def prepare_drl_data():
    """
    全自动流程核心：
    1. 扫描磁盘 -> 2. 读取内存 -> 3. 提取特征 -> 4. 释放原始大图 -> 5. 返回训练集
    """
    print("📥 [全自动模式] 开始扫描并处理数据...")

    # 1. 准备校准数据
    try:
        white = load_calib_hdr(WHITE_REF_HDR)
        dark = load_calib_hdr(DARK_REF_HDR)
        denom = (white - dark)
        denom[denom == 0] = 1e-6
    except Exception as e:
        raise FileNotFoundError(f"校准文件读取失败: {e}")

    X_list, y_list = [], []

    # 2. 扫描文件
    all_files = os.listdir(SPE_ROOT)
    spe_files = [f for f in all_files if f.lower().endswith('.spe')]
    print(f"🔎 发现 {len(spe_files)} 个光谱文件，开始内存处理...")

    for fname in spe_files:
        # 如果样本够了就停止，节省时间
        if len(X_list) * (SAMPLE_PIXELS_PER_IMAGE // 2) > MAX_TOTAL_SAMPLES:
            break

        # 路径构建
        base_name = os.path.splitext(fname)[0]
        spe_path = os.path.join(SPE_ROOT, fname)
        hdr_path = os.path.join(SPE_ROOT, base_name + ".hdr")
        json_path = os.path.join(JSON_ROOT, base_name + ".json")

        if not os.path.exists(hdr_path) or not os.path.exists(json_path):
            continue

        try:
            # 3. 加载与校准
            fix_header_byte_order(hdr_path)
            raw = envi.open(hdr_path, spe_path).load()
            if raw.shape[1] == 208:
                raw = np.transpose(raw, (0, 2, 1))

            calib = (raw.astype(np.float32) - dark) / denom
            mask = get_mask_from_json(json_path, (calib.shape[0], calib.shape[1]))

            if mask is None: continue

            # 4. 提取特征像素 (避免把整张图存入内存)
            current_X, current_y = [], []
            for m_val, target in [(1, 1), (2, 0)]:
                idx = np.where(mask == m_val)
                if len(idx[0]) > 0:
                    size = min(len(idx[0]), SAMPLE_PIXELS_PER_IMAGE // 2)
                    s_idx = np.random.choice(len(idx[0]), size=size, replace=False)
                    current_X.append(calib[idx[0][s_idx], idx[1][s_idx], :])
                    current_y.append(np.full(size, target))

            if current_X:
                X_list.append(np.concatenate(current_X))
                y_list.append(np.concatenate(current_y))
                print(f"  + 已提取: {fname}", end='\r')

            # 5. 立即释放大图内存
            del raw, calib, mask
            gc.collect()

        except Exception as e:
            print(f"\n❌ 处理出错 {fname}: {e}")

    if not X_list:
        raise ValueError("未能提取到数据！请检查路径或 JSON 标签。")

    print(f"\n✅ 数据集构建完成！总样本数: {sum(len(x) for x in X_list)}")
    return np.concatenate(X_list), np.concatenate(y_list)


def start_training():
    # === 阶段 1: 数据准备 (内存直通) ===
    X_train, y_train = prepare_drl_data()
    num_bands = X_train.shape[1]

    # === 阶段 2: 预计算指标 ===
    print("🧠 正在计算熵与互信息 (这可能需要几分钟)...")
    all_entropies = precompute_entropies(X_train)
    all_mi_scores = precompute_mutual_information(X_train, y_train)

    # === 阶段 3: 初始化 Agent ===
    agent = BandSelectionAgent(num_bands)

    print(f"\n🚀 DRL 训练启动 | 目标: 挑选 {NUM_BANDS_TO_SELECT} 个波段")
    best_bands = []
    best_reward = -float('inf')

    # === 阶段 4: 循环训练 ===
    for e in range(TOTAL_EPISODES):
        state = np.zeros(num_bands)
        selected = []
        total_r = 0

        for _ in range(NUM_BANDS_TO_SELECT):
            action = agent.get_action(state, selected)
            reward = calculate_reward(selected, action, all_entropies, all_mi_scores, alpha=ALPHA)

            next_state = state.copy()
            next_state[action] = 1
            done = (len(selected) == NUM_BANDS_TO_SELECT - 1)

            agent.remember(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            selected.append(action)
            total_r += reward

        agent.update_target_network()
        # Epsilon 衰减
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # 记录最佳结果
        if total_r > best_reward:
            best_reward = total_r
            best_bands = sorted(selected)

        if (e + 1) % 10 == 0:
            print(f"Episode: {e + 1}/{TOTAL_EPISODES}, Reward: {total_r:.4f}, Epsilon: {agent.epsilon:.2f}")

    print("\n" + "=" * 50)
    print("🏆 最优波段组合 (可以直接用于 C++):")
    print(best_bands)
    print("=" * 50)


if __name__ == "__main__":
    # GPU 显存动态分配
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass

    start_training()