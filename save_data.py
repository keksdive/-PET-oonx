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
        "spe_dir": r"D:\Train_Data\fake_img\train-PET",
        "json_dir": r"D:\Train_Data\fake_img\train-PET\fake_images"
    },
    # 2. 非 PET 文件夹 (CC) -> 将作为负样本 (背景/容易区分)
    {
        "spe_dir": r"D:\Train_Data\no_PET\CC",
        "json_dir": None
    },
    # 3. 非 PET 文件夹 (PA) -> 将作为困难负样本 (需重点加权)
    {
        "spe_dir": r"D:\Train_Data\no_PET\PA",
        "json_dir": None
    }
]

# [新增] 校准文件路径
WHITE_REF_PATH = r"D:\Train_Data\DWA\white_ref.spe"
DARK_REF_PATH = r"D:\Train_Data\DWA\dark_ref.spe"

# 输出保存路径
OUTPUT_DIR = r"D:\Processed_Result\67w-38w\procession-data"

# 标签定义：三分类逻辑 (支持困难样本挖掘)
LABEL_MAP = {
    "PET": 1,       # 正样本
    "NON_PET": 0,   # 普通负样本 (背景, CC, PP等)
    "PA": 2         # 困难负样本 (尼龙) -> 对应 Class Weight 高权重
}

# 采样参数
SAMPLES_PER_IMAGE = 4000
THRESHOLD_RATIO = 0.05
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
    """加载黑白校准文件并计算平均光谱"""
    print(f"⚪ 加载白板: {white_path}")
    print(f"⚫ 加载黑板: {dark_path}")

    def load_mean(path):
        hdr = os.path.splitext(path)[0] + ".hdr"
        repair_hdr_file(hdr)
        if not os.path.exists(path) or not os.path.exists(hdr):
            raise FileNotFoundError(f"缺失校准文件: {path}")
        img = envi.open(hdr, path).load()
        return np.mean(img, axis=(0, 1)).astype(np.float32)

    try:
        w = load_mean(white_path)
        d = load_mean(dark_path)
        return w, d
    except Exception as e:
        print(f"❌ 校准文件加载失败: {e}")
        exit()


def load_envi_image_with_calibration(hdr_path, white_ref, dark_ref):
    """加载 ENVI 图像并立即执行黑白校正"""
    try:
        repair_hdr_file(hdr_path)
        base = os.path.splitext(hdr_path)[0]
        spe_path = base + ".spe"
        if not os.path.exists(spe_path): spe_path = base + ".raw"
        if not os.path.exists(spe_path): return None

        # 1. 加载原始 RAW 数据
        img_obj = envi.open(hdr_path, spe_path)
        raw_data = np.array(img_obj.load(), dtype=np.float32)

        # 2. 维度修正
        shape = raw_data.shape
        if shape[1] < shape[2] and shape[1] in [206, 208, 224]:
            raw_data = np.transpose(raw_data, (0, 2, 1))

        H, W, B = raw_data.shape

        # 3. 黑白校正
        if white_ref.shape[0] != B:
            w_res = cv2.resize(white_ref.reshape(1, -1), (B, 1)).flatten()
            d_res = cv2.resize(dark_ref.reshape(1, -1), (B, 1)).flatten()
        else:
            w_res, d_res = white_ref, dark_ref

        denom = w_res - d_res
        denom[denom == 0] = 1e-6
        reflectance = (raw_data - d_res) / denom

        # 4. 波段对齐
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
    """计算强度并进行阈值过滤"""
    intensity = np.mean(img_data, axis=2)
    limit = np.max(intensity) * THRESHOLD_RATIO
    return intensity > limit


from scipy.signal import savgol_filter


def preprocess_spectra(pixels, use_snv=True, use_savgol=True, use_derivative=False):
    """
    综合预处理管道
    pixels: (N, Bands)
    """
    data = pixels.copy()

    # 1. Savitzky-Golay 平滑 (去噪)
    if use_savgol:
        # window_length 需根据波段间隔调整，通常 5-11 之间
        data = savgol_filter(data, window_length=9, polyorder=2, axis=1)

    # 2. 一阶导数 (可选，突出特征)
    if use_derivative:
        data = np.gradient(data, axis=1)

    # 3. 归一化 (SNV 优于 MinMax)
    if use_snv:
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = 1e-6
        data = (data - mean) / std
    else:
        # 如果仍坚持用 MinMax，建议先平滑再 MinMax
        p_min = data.min(axis=1, keepdims=True)
        p_max = data.max(axis=1, keepdims=True)
        rng = p_max - p_min
        rng[rng == 0] = 1e-6
        data = (data - p_min) / rng

    return data.astype(np.float32)


def filter_outliers(pixels, labels, purity_threshold=0.90):
    """
    基于光谱角的离群点剔除 (简单版)
    """
    clean_pixels = []
    clean_labels = []

    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        cls_pixels = pixels[idx]

        # 计算该类平均光谱 (Centroid)
        centroid = np.mean(cls_pixels, axis=0)
        norm_centroid = np.linalg.norm(centroid)

        # 计算余弦相似度 (Cosine Similarity)
        # A . B / (|A| * |B|)
        norms = np.linalg.norm(cls_pixels, axis=1)
        dots = np.dot(cls_pixels, centroid)
        sims = dots / (norms * norm_centroid + 1e-6)

        # 保留相似度高的纯净像素
        mask = sims >= purity_threshold
        clean_pixels.append(cls_pixels[mask])
        clean_labels.append(labels[idx][mask])

        print(f"   🧹 Class {lbl}: 剔除 {len(idx) - np.sum(mask)} 个离群杂质像素")

    return np.vstack(clean_pixels), np.concatenate(clean_labels)



def determine_label(path_string):
    """
    [修改] 核心标签判断逻辑
    PET -> 1
    PA (尼龙) -> 2 (独立类别)
    其他非PET -> 0
    """
    path_upper = path_string.upper()

    # 1. 优先判断 PET
    if "PET" in path_upper and "NO_PET" not in path_upper and "NO-PET" not in path_upper:
        return LABEL_MAP["PET"], "PET"

    # 2. [新增] 专门判断 PA (尼龙)
    # 只要路径或文件名中包含 PA，就归为类别 2
    if "PA" in path_upper:
        return LABEL_MAP["PA"], "PA"

    # 3. 其他负样本判断
    # 注意：PA 已经从这个列表中移除，或者上面的 if "PA" 会先拦截
    negative_keys = ["CC", "PP", "醋酸", "OTHER", "NO_PET", "NO-PET"]
    for key in negative_keys:
        if key in path_upper:
            return LABEL_MAP["NON_PET"], "NON_PET"

    return None, None


def process_and_save_data():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print("📥 正在加载黑白校准文件...")
    white_ref, dark_ref = load_calibration_data(WHITE_REF_PATH, DARK_REF_PATH)

    all_pixels, all_labels = [], []
    # [修改] 增加类别 2 的统计槽位
    stats = {0: 0, 1: 0, 2: 0}
    total_files = 0

    print(f"🚀 [专家模式] 开始处理数据 (已启用 PA 独立分类)")
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

            img_data = load_envi_image_with_calibration(hdr_path, white_ref, dark_ref)
            if img_data is None: continue

            # 获取掩膜
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

            valid_pixels = img_data[fg_mask]

            if len(valid_pixels) > 0:
                if len(valid_pixels) > SAMPLES_PER_IMAGE:
                    indices = np.random.choice(len(valid_pixels), SAMPLES_PER_IMAGE, replace=False)
                    valid_pixels = valid_pixels[indices]

                norm_pixels = min_max_normalize(valid_pixels)

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

    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    print("-" * 30)
    print(f"✅ 完成! 总文件: {total_files}")
    print(f"📊 正样本   (PET, Label 1): {stats[1]}")
    print(f"📊 普通负样 (CC/Label 0):   {stats[0]}")
    print(f"📊 困难负样 (PA/Label 2):   {stats[2]} <--- 确认这里有数据!")

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    print(f"💾 已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    process_and_save_data()