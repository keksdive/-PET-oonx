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
        "spe_dir": r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET",
        "json_dir": r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET\fake_images"
    },
    # 2. 非 PET 文件夹 (CC)
    {
        "spe_dir": r"I:\SPEDATA\高谱相机数据集\训练集\no_PET\CC",
        "json_dir": None
    },
    # 3. 非 PET 文件夹 (PA)
    {
        "spe_dir": r"I:\SPEDATA\高谱相机数据集\训练集\no_PET\PA",
        "json_dir": None
    }
]

# 输出保存路径
OUTPUT_DIR = r"D:\DRL\DRL1\.gitignore\data"

# 标签定义
LABEL_MAP = {
    "PET": 1,
    "CC": 2,
    "PA": 3,
    "PP": 4,
    "OTHER": 5,
    "醋酸": 2  # 中文兼容
}

# 采样参数
SAMPLES_PER_IMAGE = 2000
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
            # print(f"🔧 修复头文件: {os.path.basename(hdr_path)}")
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

        # 波段对齐 (206 -> 208)
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
    B = img_data.shape[2]
    start, end = 10, B - 10
    intensity = np.mean(img_data[:, :, start:end], axis=2)
    return intensity > (np.max(intensity) * THRESHOLD_RATIO)


def determine_label(path_string):
    """
    [核心修复] 更智能的标签判断逻辑
    1. 优先匹配具体的非PET材质 (CC, PA, PP)
    2. 只有在不包含 'no_PET' 的情况下，才匹配 PET
    """
    path_upper = path_string.upper()

    # 1. 优先检查具体材质 (防止被 no_PET 中的 PET 关键字误导)
    for key in ["CC", "PA", "PP", "醋酸", "OTHER"]:
        if key in path_upper:
            return LABEL_MAP.get(key, LABEL_MAP.get("OTHER")), key

    # 2. 检查 PET，但必须排除 no_PET 文件夹
    if "PET" in path_upper:
        # 如果路径里有 no_PET 或 no-PET，这绝对不是 PET 类别
        if "NO_PET" in path_upper or "NO-PET" in path_upper:
            return None, None
        return LABEL_MAP["PET"], "PET"

    return None, None


def process_and_save_data():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    all_pixels, all_labels = [], []
    stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_files = 0

    print(f"🚀 开始处理 {len(DATASETS)} 个数据源...")

    for ds_config in DATASETS:
        spe_dir, json_dir = ds_config["spe_dir"], ds_config.get("json_dir")
        if not os.path.exists(spe_dir): continue

        hdr_files = glob.glob(os.path.join(spe_dir, "**", "*.hdr"), recursive=True)
        print(f"📂 扫描: {spe_dir} ({len(hdr_files)} files)")

        for idx, hdr_path in enumerate(hdr_files):
            if "ref" in os.path.basename(hdr_path).lower(): continue

            full_path_str = hdr_path  # 使用全路径进行判断
            label_id, label_name = determine_label(full_path_str)
            if label_id is None: continue

            img_data = load_envi_image(hdr_path)
            if img_data is None: continue

            fg_mask = None
            mode = "Auto"
            if json_dir:
                base = os.path.splitext(os.path.basename(hdr_path))[0]
                jp = os.path.join(json_dir, base + ".json")
                if os.path.exists(jp):
                    fg_mask = get_mask_from_json(jp, img_data.shape)
                    mode = "JSON"

            if fg_mask is None: fg_mask = get_mask_from_threshold(img_data)

            # 采样
            for m, lid in [(fg_mask, label_id), (~fg_mask, 0)]:
                pix = img_data[m]
                if len(pix) > SAMPLES_PER_IMAGE:
                    pix = pix[np.random.choice(len(pix), SAMPLES_PER_IMAGE, replace=False)]
                if len(pix) > 0:
                    all_pixels.append(pix)
                    all_labels.append(np.full(len(pix), lid, dtype=np.int32))
                    stats[lid] += len(pix)

            total_files += 1
            if idx % 20 == 0:
                print(f"   [{idx + 1}] {os.path.basename(hdr_path):<20} | 🏷️ {label_name}({label_id}) | ⚙️ {mode}")

    if not all_pixels: return print("❌ 无数据")

    print("\n📦 合并数据...")
    X = np.vstack(all_pixels)
    y = np.concatenate(all_labels)
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    print("-" * 30)
    print(f"✅ 完成! 总文件: {total_files}")
    for k, v in stats.items():
        if v > 0: print(f"  Label {k}: {v}")

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)


if __name__ == "__main__":
    process_and_save_data()