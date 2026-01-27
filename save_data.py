import os
import numpy as np
import spectral.io.envi as envi
import glob
import json
import cv2
import random
import time
from scipy.signal import savgol_filter
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ================= ⚙️ 1. 多源数据集配置 (专家模式) =================

DATASETS = [
    # --- 正样本 (PET) ---
    {
        "spe_dir": r"E:\SPEDATA\高谱相机数据集\训练集\清洗数据测试\PET",
        "json_dir": None,
        "label_id": 1,  # PET
        "name": "PET"
    },
    # --- 负样本 (CC - 碳酸钙) ---
    {
        "spe_dir": r"E:\SPEDATA\高谱相机数据集\训练集\清洗数据测试\CC",
        "json_dir": None,
        "label_id": 3,  # CC
        "name": "CC"
    },
    # --- 负样本 (PA - 尼龙) ---
    {
        "spe_dir": r"E:\SPEDATA\高谱相机数据集\训练集\清洗数据测试\PA",
        "json_dir": None,
        "label_id": 2,  # PA
        "name": "PA"
    }
]

# 校准文件路径
WHITE_REF_PATH = r"E:\SPEDATA\高谱相机数据集\DWA\white_ref.spe"
DARK_REF_PATH = r"E:\SPEDATA\高谱相机数据集\DWA\dark_ref.spe"

# 输出保存路径
OUTPUT_DIR = r"M:\PROcess-data\material-feature"

# 采样与清洗参数
SAMPLES_PER_IMAGE = 5000
TARGET_BANDS = 208
# PURITY_THRESHOLD 已被移除，因为我们不再使用余弦相似度过滤
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3
BINARY_THRESHOLD = 0.25  # ✅ 建议调回 0.15 (0.25 可能太高)，这是唯一的过滤门槛


# ================= 🛠️ 2. 核心算法工具库 =================

def apply_snv(spectra):
    """标准正态变量变换 (SNV)"""
    spectra = spectra.astype(np.float32)
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    std[std == 0] = 1e-6
    return (spectra - mean) / std


def apply_derivative(spectra, window=11, poly=3):
    """Savitzky-Golay 导数"""
    return savgol_filter(spectra, window_length=window, polyorder=poly, deriv=1, axis=1)


def repair_hdr_file(hdr_path):
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
    def load_mean(path):
        hdr = os.path.splitext(path)[0] + ".hdr"
        repair_hdr_file(hdr)
        img = envi.open(hdr, path).load()
        return np.mean(img, axis=(0, 1)).astype(np.float32)

    w = load_mean(white_path)
    d = load_mean(dark_path)
    return w, d


def load_envi_image_reflectance(hdr_path, white_ref, dark_ref):
    try:
        repair_hdr_file(hdr_path)
        base = os.path.splitext(hdr_path)[0]
        spe_path = base + ".spe"
        if not os.path.exists(spe_path): spe_path = base + ".raw"
        img_obj = envi.open(hdr_path, spe_path)
        raw_data = np.array(img_obj.load(), dtype=np.float32)
        if raw_data.shape[1] < raw_data.shape[2] and raw_data.shape[1] in [206, 208, 224]:
            raw_data = np.transpose(raw_data, (0, 2, 1))
        H, W, B = raw_data.shape
        w_res = cv2.resize(white_ref.reshape(1, -1), (B, 1)).flatten() if white_ref.shape[0] != B else white_ref
        d_res = cv2.resize(dark_ref.reshape(1, -1), (B, 1)).flatten() if dark_ref.shape[0] != B else dark_ref
        denom = w_res - d_res
        denom[denom == 0] = 1e-6
        reflectance = (raw_data - d_res) / denom
        if TARGET_BANDS is not None and B != TARGET_BANDS:
            flat = reflectance.reshape(-1, B)
            flat_resized = cv2.resize(flat, (TARGET_BANDS, H * W), interpolation=cv2.INTER_LINEAR)
            reflectance = flat_resized.reshape(H, W, TARGET_BANDS)
        return reflectance
    except Exception as e:
        print(f"❌ Error {os.path.basename(hdr_path)}: {e}")
        return None


def generate_intensity_mask(img_data, threshold=0.15):
    """
    [核心过滤逻辑] 基于二值化亮度阈值的掩膜生成
    只要像素够亮，就认为是材质。不关心它的光谱长什么样。
    """
    try:
        # 1. 计算全波段平均强度
        intensity = np.mean(img_data, axis=2)

        # 2. 二值化分割
        mask = intensity > threshold

        # 3. 形态学处理 (去噪 + 填补空洞)
        mask = mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 去除背景小白点
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填补材质内部孔洞

        return mask.astype(bool)
    except:
        return (np.mean(img_data, axis=2) > threshold)


def get_mask_combined(json_path, img_data):
    H, W = img_data.shape[:2]

    # === 仅使用二值化亮度掩膜 ===
    auto_mask = generate_intensity_mask(img_data, threshold=BINARY_THRESHOLD)

    json_mask = None
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            j_mask = np.zeros((H, W), dtype=np.uint8)
            for shape in data.get('shapes', []):
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(j_mask, [points], 1)
            json_mask = j_mask.astype(bool)
        except:
            pass

    if json_mask is not None:
        return json_mask & auto_mask
    else:
        return auto_mask


def generate_cleaning_report(X, y, label_names, output_dir):
    """
    生成混淆矩阵报告
    注意：这里的'可分性'是指特征空间中的物理距离，不是模型的最终性能。
    """
    print("\n📊 [评估] 正在生成材质可分性混淆矩阵...")
    unique_labels = np.unique(y)
    centroids = []
    for label in unique_labels:
        centroids.append(np.mean(X[y == label], axis=0))
    centroids = np.array(centroids)

    sim_matrix = cosine_similarity(X, centroids)
    y_pred = unique_labels[np.argmax(sim_matrix, axis=1)]

    cm = confusion_matrix(y, y_pred)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.2%', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Material Separability Confusion Matrix (No Purity Filter)')
    plt.ylabel('True Material')
    plt.xlabel('Predicted (Nearest Centroid)')
    plot_path = os.path.join(output_dir, "cleaning_confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()

    report = classification_report(y, y_pred, target_names=label_names)
    with open(os.path.join(output_dir, "cleaning_report.txt"), "w", encoding='utf-8') as f:
        f.write("=== Data Cleaning Report (Intensity Mask Only) ===\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Binary Threshold: {BINARY_THRESHOLD}\n")
        f.write("Note: Cosine similarity filtering (Purity Check) has been disabled.\n\n")
        f.write(report)

    print(f"✅ 混淆矩阵已保存至: {plot_path}")
    print(f"✅ 文本报告已保存至: cleaning_report.txt")


# ================= 🚀 3. 主处理流程 =================

def process_and_save_data():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    white_ref, dark_ref = load_calibration_data(WHITE_REF_PATH, DARK_REF_PATH)

    # 原始数据缓存
    raw_data_buffer = {}

    # === 背景采集计数器 ===
    MAX_BG_SAMPLES = 10000
    current_bg_count = 0
    bg_data_buffer = []

    print(f"\n🔄 [阶段1] 扫描并提取原始像素...")
    for ds_config in DATASETS:
        label_id = ds_config["label_id"]
        label_name = ds_config["name"]
        if label_id not in raw_data_buffer: raw_data_buffer[label_id] = []

        print(f"   📂 正在处理: {label_name}...")
        hdr_files = glob.glob(os.path.join(ds_config["spe_dir"], "**", "*.hdr"), recursive=True)

        for hdr_path in hdr_files:
            if "ref" in os.path.basename(hdr_path).lower(): continue
            img_data = load_envi_image_reflectance(hdr_path, white_ref, dark_ref)
            if img_data is None: continue

            # --- 1. 获取材质掩膜 ---
            final_material_mask = get_mask_combined(None, img_data)  # JSON 逻辑已内嵌

            # 提取材质像素
            valid_pixels = img_data[final_material_mask]
            if len(valid_pixels) > SAMPLES_PER_IMAGE:
                valid_pixels = valid_pixels[np.random.choice(len(valid_pixels), SAMPLES_PER_IMAGE, replace=False)]
            if len(valid_pixels) > 0: raw_data_buffer[label_id].append(valid_pixels)

            # --- 2. 提取背景像素 ---
            if current_bg_count < MAX_BG_SAMPLES:
                intensity = np.mean(img_data, axis=2)
                # 背景 = (非材质) & (亮度 > 0.01)
                bg_mask = (~final_material_mask) & (intensity > 0.01)

                bg_pixels = img_data[bg_mask]
                if len(bg_pixels) > 0:
                    n_needed = MAX_BG_SAMPLES - current_bg_count
                    n_take = min(len(bg_pixels), 1000, n_needed)

                    selected_bg = bg_pixels[np.random.choice(len(bg_pixels), n_take, replace=False)]
                    bg_data_buffer.append(selected_bg)
                    current_bg_count += len(selected_bg)

    if len(bg_data_buffer) > 0:
        raw_data_buffer[0] = bg_data_buffer
        DATASETS.append({"label_id": 0, "name": "Background"})

    print(f"\n🧹 [阶段2] 数据清洗与特征工程 (已禁用余弦相似度过滤)...")
    final_X, final_y, names = [], [], []
    for label_id, pixel_list in raw_data_buffer.items():
        if not pixel_list: continue
        all_pixels = np.vstack(pixel_list)
        label_name = [d['name'] for d in DATASETS if d['label_id'] == label_id][0]

        # 1. SNV 处理
        snv_pixels = apply_snv(all_pixels)

        # 2. ❌ 移除 filter_impurities 调用 ❌
        # 我们假设阶段1的掩膜已经足够准确，不再根据光谱形状剔除样本
        clean_snv_pixels = snv_pixels

        print(f"   ✅ [{label_name}] 保留 {len(clean_snv_pixels)} 样本 (不进行光谱形状清洗)")

        # 3. 导数与特征堆叠
        deriv_pixels = apply_derivative(clean_snv_pixels, window=SAVGOL_WINDOW, poly=SAVGOL_POLY)
        stacked_features = np.concatenate([clean_snv_pixels, deriv_pixels], axis=1)
        final_X.append(stacked_features)
        final_y.append(np.full(len(stacked_features), label_id, dtype=np.int32))
        names.append(label_name)

    if not final_X: return
    X, y = np.vstack(final_X), np.concatenate(final_y)
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    print(f"\n📊 最终统计: {len(y)} 样本, {X.shape[1]} 维度")
    generate_cleaning_report(X, y, names, OUTPUT_DIR)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    print(f"💾 数据已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_and_save_data()