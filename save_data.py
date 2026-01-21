import os
import numpy as np
import spectral.io.envi as envi
import glob
import json
import cv2
import random

# ================= ⚙️ 多源数据集配置区域 =================
DATASETS = [
    # 1. PET 文件夹
    {
        "spe_dir": r"E:\SPEDATA\高谱相机数据集\训练集\置信度大于90%PET",
        "json_dir": None
    },
    # 2. 非 PET 文件夹 (CC)
    {
        "spe_dir": r"E:\SPEDATA\高谱相机数据集\训练集\no_PET\CC",
        "json_dir": None
    },
    # 3. 非 PET 文件夹 (PA)
    {
        "spe_dir": r"E:\SPEDATA\高谱相机数据集\训练集\no_PET\PA",
        "json_dir": None
    }
]

# 输出保存路径
OUTPUT_DIR = r"E:\SPEDATA\NP_newdata"

# 标签定义
LABEL_MAP = {
    "PET": 1,
    "CC": 2,
    "PA": 3,
    "PP": 4,
    "OTHER": 5,
    "醋酸": 2
}

# 采样参数
SAMPLES_PER_IMAGE = 3000  # 增加采样数，因为我们现在丢弃了背景
THRESHOLD_RATIO = 0.15  # 【严格过滤】低于最大亮度 15% 的直接丢弃
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


def load_envi_image(hdr_path):
    """加载并对齐 ENVI 图像"""
    try:
        repair_hdr_file(hdr_path)
        base = os.path.splitext(hdr_path)[0]
        spe_path = base + ".spe"
        if not os.path.exists(spe_path): spe_path = base + ".raw"
        if not os.path.exists(spe_path): return None

        img_obj = envi.open(hdr_path, spe_path)
        img_data = np.array(img_obj.load(), dtype=np.float32)

        # 维度修正 (H, W, B)
        shape = img_data.shape
        if shape[1] < shape[2] and shape[1] in [206, 208]:
            img_data = np.transpose(img_data, (0, 2, 1))

        # 波段对齐
        H, W, C = img_data.shape
        if TARGET_BANDS is not None and C != TARGET_BANDS:
            flat = img_data.reshape(-1, C)
            flat_resized = cv2.resize(flat, (TARGET_BANDS, H * W), interpolation=cv2.INTER_LINEAR)
            img_data = flat_resized.reshape(H, W, TARGET_BANDS)

        return img_data
    except Exception as e:
        print(f"❌ 加载失败 {os.path.basename(hdr_path)}: {e}")
        return None


def get_mask_from_json(json_path, image_shape):
    """优先使用 JSON 标注（如果有）"""
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
    """【功能1 & 3】计算强度并进行严格阈值过滤"""
    # 计算平均强度
    intensity = np.mean(img_data, axis=2)
    # 动态阈值：最大强度的 15%
    limit = np.max(intensity) * THRESHOLD_RATIO
    # 生成掩膜：只有大于阈值的才是 True
    return intensity > limit


def min_max_normalize(pixels):
    """【功能2】Min-Max 归一化 (针对像素级)"""
    # pixels shape: (N, Bands)
    # axis=1 表示对每个像素自身的波段进行归一化
    p_min = pixels.min(axis=1, keepdims=True)
    p_max = pixels.max(axis=1, keepdims=True)

    # 避免除以0
    range_val = p_max - p_min
    range_val[range_val == 0] = 1e-6

    return (pixels - p_min) / range_val


def determine_label(path_string):
    path_upper = path_string.upper()
    for key in ["CC", "PA", "PP", "醋酸", "OTHER"]:
        if key in path_upper:
            return LABEL_MAP.get(key, LABEL_MAP.get("OTHER")), key
    if "PET" in path_upper:
        if "NO_PET" in path_upper or "NO-PET" in path_upper:
            return None, None
        return LABEL_MAP["PET"], "PET"
    return None, None


def process_and_save_data():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    all_pixels, all_labels = [], []
    stats = {k: 0 for k in LABEL_MAP.values()}  # 统计所有标签
    total_files = 0

    print(f"🚀 开始处理 {len(DATASETS)} 个数据源...")

    for ds_config in DATASETS:
        spe_dir, json_dir = ds_config["spe_dir"], ds_config.get("json_dir")
        if not os.path.exists(spe_dir): continue

        hdr_files = glob.glob(os.path.join(spe_dir, "**", "*.hdr"), recursive=True)
        print(f"📂 扫描: {spe_dir} ({len(hdr_files)} files)")

        for idx, hdr_path in enumerate(hdr_files):
            if "ref" in os.path.basename(hdr_path).lower(): continue

            full_path_str = hdr_path
            label_id, label_name = determine_label(full_path_str)
            if label_id is None: continue

            img_data = load_envi_image(hdr_path)
            if img_data is None: continue

            # 1. 获取有效区域掩膜
            fg_mask = None
            mode = "Threshold"

            # 如果有 JSON，先尝试 JSON，再用阈值过滤 JSON 选区内的杂色
            if json_dir:
                base = os.path.splitext(os.path.basename(hdr_path))[0]
                jp = os.path.join(json_dir, base + ".json")
                if os.path.exists(jp):
                    json_mask = get_mask_from_json(jp, img_data.shape)
                    if json_mask is not None:
                        # 即使有 JSON，也要再叠一层亮度过滤，去掉标注框里的黑色背景
                        thresh_mask = get_mask_from_threshold(img_data)
                        fg_mask = json_mask & thresh_mask
                        mode = "JSON+Threshold"

            # 如果没有 JSON，直接用阈值
            if fg_mask is None:
                fg_mask = get_mask_from_threshold(img_data)
                mode = "Auto-Threshold"

            # 2. 【严格过滤】提取像素
            # 只提取 Mask 为 True 的部分 (即 > 15% 亮度的部分)
            # 丢弃所有 Mask 为 False 的部分 (背景)
            valid_pixels = img_data[fg_mask]

            if len(valid_pixels) > 0:
                # 随机采样，防止数据量过大
                if len(valid_pixels) > SAMPLES_PER_IMAGE:
                    indices = np.random.choice(len(valid_pixels), SAMPLES_PER_IMAGE, replace=False)
                    valid_pixels = valid_pixels[indices]

                # 3. 【归一化】执行 Min-Max 归一化
                # 将数据映射到 0-1，消除光强影响
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
        print("❌ 无数据，请检查路径或阈值设置是否过高。")
        return

    print("\n📦 合并数据...")
    X = np.vstack(all_pixels)
    y = np.concatenate(all_labels)

    # 打乱数据
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    print("-" * 30)
    print(f"✅ 完成! 总文件: {total_files}")
    for k, v in stats.items():
        if v > 0: print(f"  Label {k}: {v}")
    print(f"📊 数据形状: X={X.shape}, y={y.shape}")
    print(f"📉 数据范围: Min={X.min():.4f}, Max={X.max():.4f} (应为 0~1)")

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    print(f"💾 已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    process_and_save_data()