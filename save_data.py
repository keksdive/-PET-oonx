import os
import numpy as np
import spectral.io.envi as envi
import glob
import json
import cv2
import random

# ================= ⚙️ 多源数据集配置区域 =================
DATASETS = [
    # 1. PET 文件夹 (正样本)
    {
        "spe_dir": r"E:\SPEDATA\高谱相机数据集\训练集\置信度大于90%PET",
        "json_dir": None
    },
    # 2. 非 PET 文件夹 (CC) -> 将作为负样本
    {
        "spe_dir": r"E:\SPEDATA\高谱相机数据集\训练集\no_PET\CC",
        "json_dir": None
    },
    # 3. 非 PET 文件夹 (PA) -> 将作为负样本
    {
        "spe_dir": r"E:\SPEDATA\高谱相机数据集\训练集\no_PET\PA",
        "json_dir": None
    }
]

# [新增] 校准文件路径 (请确认路径是否正确)
WHITE_REF_PATH = r"E:\SPEDATA\高谱相机数据集\DWA\white_ref.spe"
DARK_REF_PATH = r"E:\SPEDATA\高谱相机数据集\DWA\dark_ref.spe"

# 输出保存路径
OUTPUT_DIR = r"E:\SPEDATA\NP_new1.0.2"

# 标签定义：二分类逻辑
LABEL_MAP = {
    "PET": 1,
    "NON_PET": 0
}

# 采样参数
SAMPLES_PER_IMAGE = 3000
THRESHOLD_RATIO = 0.15
TARGET_BANDS = 208  # 强制对齐波段数


# =======================================================

def repair_hdr_file(hdr_path):
    """自动修复缺少的 byte order"""
    try:
        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        content = "".join(lines).lower()
        if "byte order" not in content:
            lines.append("\nbyte order = 0\n")
            with open(hdr_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
    except:
        pass


def load_calibration_data(white_path, dark_path):
    """
    [新增] 加载黑白校准文件并计算平均光谱
    返回: (white_mean, dark_mean) 维度为 (Bands,)
    """
    print(f"⚪ 加载白板: {white_path}")
    print(f"⚫ 加载黑板: {dark_path}")

    def load_mean(path):
        hdr = os.path.splitext(path)[0] + ".hdr"
        repair_hdr_file(hdr)
        if not os.path.exists(path) or not os.path.exists(hdr):
            raise FileNotFoundError(f"缺失校准文件: {path}")
        img = envi.open(hdr, path).load()
        # 计算空间维度的平均值，得到纯光谱向量
        return np.mean(img, axis=(0, 1)).astype(np.float32)

    try:
        w = load_mean(white_path)
        d = load_mean(dark_path)
        return w, d
    except Exception as e:
        print(f"❌ 校准文件加载失败: {e}")
        exit()


def load_envi_image_with_calibration(hdr_path, white_ref, dark_ref):
    """
    [修改] 加载 ENVI 图像并立即执行黑白校正
    Reflectance = (Raw - Dark) / (White - Dark)
    """
    try:
        repair_hdr_file(hdr_path)
        base = os.path.splitext(hdr_path)[0]
        spe_path = base + ".spe"
        if not os.path.exists(spe_path): spe_path = base + ".raw"
        if not os.path.exists(spe_path): return None

        # 1. 加载原始 RAW 数据 (DN值)
        img_obj = envi.open(hdr_path, spe_path)
        raw_data = np.array(img_obj.load(), dtype=np.float32)

        # 2. 维度修正 (确保是 H, W, B)
        shape = raw_data.shape
        if shape[1] < shape[2] and shape[1] in [206, 208, 224]:
            raw_data = np.transpose(raw_data, (0, 2, 1))

        H, W, B = raw_data.shape

        # 3. [核心] 执行黑白校正 (反射率计算)
        # 自动适配校准文件的波段数 (防止因 208 vs 224 导致的 crash)
        if white_ref.shape[0] != B:
            # 如果波段不匹配，简单线性插值校准数据到图像的波段数
            # 注意：这是为了防止报错的兜底策略，理想情况下应一致
            w_res = cv2.resize(white_ref.reshape(1, -1), (B, 1)).flatten()
            d_res = cv2.resize(dark_ref.reshape(1, -1), (B, 1)).flatten()
        else:
            w_res, d_res = white_ref, dark_ref

        denom = w_res - d_res
        denom[denom == 0] = 1e-6  # 防止除零

        # 利用广播机制计算反射率
        reflectance = (raw_data - d_res) / denom

        # 裁剪异常值 (0~1 之外的通常是噪声)
        # reflectance = np.clip(reflectance, 0, 1.5) # 可选，暂不强制 clip，保留高光特征

        # 4. 波段对齐 (Resize 到 TARGET_BANDS)
        if TARGET_BANDS is not None and B != TARGET_BANDS:
            flat = reflectance.reshape(-1, B)
            flat_resized = cv2.resize(flat, (TARGET_BANDS, H * W), interpolation=cv2.INTER_LINEAR)
            reflectance = flat_resized.reshape(H, W, TARGET_BANDS)

        return reflectance

    except Exception as e:
        print(f"❌ 加载或校正失败 {os.path.basename(hdr_path)}: {e}")
        return None


def get_mask_from_json(json_path, image_shape):
    """优先使用 JSON 标注"""
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data.get('shapes', []):
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        return mask.astype(bool)
    except:
        return None


def get_mask_from_threshold(img_data):
    """计算强度并进行严格阈值过滤 (基于反射率)"""
    intensity = np.mean(img_data, axis=2)
    # 注意：反射率通常在 0~1 之间，所以阈值逻辑依然适用
    # 但如果反光很强可能 >1，取 max * ratio 依然是稳健的
    limit = np.max(intensity) * THRESHOLD_RATIO
    return intensity > limit


def min_max_normalize(pixels):
    """
    Min-Max 归一化 (针对像素级)
    虽然已经是反射率了，但为了输入神经网络，再次归一化到 0-1 也是常见的做法
    """
    p_min = pixels.min(axis=1, keepdims=True)
    p_max = pixels.max(axis=1, keepdims=True)
    range_val = p_max - p_min
    range_val[range_val == 0] = 1e-6
    return (pixels - p_min) / range_val


def determine_label(path_string):
    path_upper = path_string.upper()

    if "PET" in path_upper and "NO_PET" not in path_upper and "NO-PET" not in path_upper:
        return LABEL_MAP["PET"], "PET"

    negative_keys = ["CC", "PA", "PP", "醋酸", "OTHER", "NO_PET", "NO-PET"]
    for key in negative_keys:
        if key in path_upper:
            return LABEL_MAP["NON_PET"], "NON_PET"

    return None, None


def process_and_save_data():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 1. [新增] 预加载校准数据
    print("📥 正在加载黑白校准文件...")
    white_ref, dark_ref = load_calibration_data(WHITE_REF_PATH, DARK_REF_PATH)

    all_pixels, all_labels = [], []
    stats = {0: 0, 1: 0}
    total_files = 0

    print(f"🚀 [专家模式] 开始处理数据 (已启用黑白辐射校正)")
    print(f"📂 正在扫描 {len(DATASETS)} 个数据源...")

    for ds_config in DATASETS:
        spe_dir, json_dir = ds_config["spe_dir"], ds_config.get("json_dir")
        if not os.path.exists(spe_dir): continue

        hdr_files = glob.glob(os.path.join(spe_dir, "**", "*.hdr"), recursive=True)

        for idx, hdr_path in enumerate(hdr_files):
            if "ref" in os.path.basename(hdr_path).lower(): continue

            full_path_str = hdr_path
            label_id, label_name = determine_label(full_path_str)

            if label_id is None: continue

            # [修改] 调用带校正的加载函数
            img_data = load_envi_image_with_calibration(hdr_path, white_ref, dark_ref)
            if img_data is None: continue

            # 1. 获取有效区域掩膜
            fg_mask = None
            mode = "Threshold"

            if json_dir:
                base = os.path.splitext(os.path.basename(hdr_path))[0]
                jp = os.path.join(json_dir, base + ".json")
                if os.path.exists(jp):
                    json_mask = get_mask_from_json(jp, img_data.shape)
                    if json_mask is not None:
                        thresh_mask = get_mask_from_threshold(img_data)
                        fg_mask = json_mask & thresh_mask
                        mode = "JSON+Threshold"

            if fg_mask is None:
                fg_mask = get_mask_from_threshold(img_data)
                mode = "Auto-Threshold"

            # 2. 提取像素
            valid_pixels = img_data[fg_mask]

            if len(valid_pixels) > 0:
                if len(valid_pixels) > SAMPLES_PER_IMAGE:
                    indices = np.random.choice(len(valid_pixels), SAMPLES_PER_IMAGE, replace=False)
                    valid_pixels = valid_pixels[indices]

                # 3. 归一化 (反射率已经是物理量，但为了神经网络稳定性，再次归一化)
                norm_pixels = min_max_normalize(valid_pixels)

                # 4. 保存
                all_pixels.append(norm_pixels)
                all_labels.append(np.full(len(norm_pixels), label_id, dtype=np.int32))
                stats[label_id] += len(norm_pixels)

            total_files += 1
            if idx % 20 == 0:
                print(
                    f"   [{idx + 1}] {os.path.basename(hdr_path):<20} | 🏷️ {label_name}({label_id}) | ⚙️ {mode} | 样本数: {len(valid_pixels)}")

    if not all_pixels:
        print("❌ 无数据，请检查路径。")
        return

    print("\n📦 合并数据...")
    X = np.vstack(all_pixels)
    y = np.concatenate(all_labels)

    # 打乱数据
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    print("-" * 30)
    print(f"✅ 完成! 总文件: {total_files}")
    print(f"📊 正样本 (PET, Label 1): {stats[1]}")
    print(f"📊 负样本 (CC/PA/杂波, Label 0): {stats[0]}")

    # 检查数值范围
    print(f"📉 数据范围: Min={X.min():.4f}, Max={X.max():.4f}")
    if X.max() > 1.0 or X.min() < 0.0:
        print("⚠️ 警告: 数据范围超出 0-1，可能 Min-Max 归一化有误或原始反射率异常高")

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    print(f"💾 已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    process_and_save_data()