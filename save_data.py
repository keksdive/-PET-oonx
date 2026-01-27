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
        "spe_dir": r"D:\Train_Data\fake_img\train-PET",
        "json_dir": r"D:\Train_Data\fake_img\train-PET\fake_images",
        "label_id": 1,  # PET
        "name": "PET"
    },
    # --- 负样本 (CC - 碳酸钙) ---
    {
        "spe_dir": r"D:\Train_Data\no_PET\CC",
        "json_dir": None,
        "label_id": 3,  # CC
        "name": "CC"
    },
    # --- 负样本 (PA - 尼龙) ---
    {
        "spe_dir": r"D:\Train_Data\no_PET\PA",
        "json_dir": None,
        "label_id": 2,  # PA
        "name": "PA"
    }
]

# 校准文件路径
WHITE_REF_PATH = r"D:\Train_Data\DWA\white_ref.spe"
DARK_REF_PATH = r"D:\Train_Data\DWA\dark_ref.spe"

# 输出保存路径
OUTPUT_DIR = r"D:\Processed_Result\material-feature"

# 采样与清洗参数
SAMPLES_PER_IMAGE = 5000
TARGET_BANDS = 208
PURITY_THRESHOLD = 0.80
SAVGOL_WINDOW = 11
SAVGOL_POLY = 3


# ================= 🛠️ 2. 核心算法工具库 =================

def apply_snv(spectra):
    spectra = spectra.astype(np.float32)
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    std[std == 0] = 1e-6
    return (spectra - mean) / std


def apply_derivative(spectra, window=11, poly=3):
    return savgol_filter(spectra, window_length=window, polyorder=poly, deriv=1, axis=1)


def filter_impurities(pixels, label_name, threshold=0.95):
    if len(pixels) == 0:
        return pixels
    mean_spectrum = np.mean(pixels, axis=0).reshape(1, -1)
    similarities = cosine_similarity(pixels, mean_spectrum)
    mask = similarities.flatten() >= threshold
    clean_pixels = pixels[mask]
    drop_rate = (1 - len(clean_pixels) / len(pixels)) * 100
    print(f"   🧹 [{label_name}] 清洗: 原始 {len(pixels)} -> 保留 {len(clean_pixels)} (剔除率 {drop_rate:.1f}%)")
    return clean_pixels


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


def get_mask_combined(json_path, img_data):
    H, W = img_data.shape[:2]
    intensity = np.mean(img_data, axis=2)
    thresh_mask = intensity > 0.10
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
    return (json_mask & thresh_mask) if json_mask is not None else thresh_mask


def generate_cleaning_report(X, y, label_names, output_dir):
    """验证处理后各类材质的物理可分性"""
    print("\n📊 [评估] 正在生成材质可分性混淆矩阵...")
    unique_labels = np.unique(y)
    centroids = [np.mean(X[y == label], axis=0) for label in unique_labels]
    centroids = np.array(centroids)
    sim_matrix = cosine_similarity(X, centroids)
    y_pred = unique_labels[np.argmax(sim_matrix, axis=1)]

    cm = confusion_matrix(y, y_pred)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.2%', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Material Separability Confusion Matrix (After Cleaning)')
    plt.ylabel('True Material')
    plt.xlabel('Predicted (Nearest Centroid)')
    plot_path = os.path.join(output_dir, "cleaning_confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()

    report = classification_report(y, y_pred, target_names=label_names)
    with open(os.path.join(output_dir, "cleaning_report.txt"), "w", encoding='utf-8') as f:
        f.write("=== Data Cleaning & Separability Report ===\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report)
    print(f"✅ 报告已保存至: {output_dir}")



def generate_cleaning_report(X, y, label_names, output_dir):
    """
    [新增功能] 生成清洗后数据的混淆矩阵和分类报告
    验证 SNV + 导数处理后，各类材质的物理可分性
    """
    print("\n📊 [评估] 正在生成材质可分性混淆矩阵...")

    # 1. 计算每个类别的质心 (Centroids)
    unique_labels = np.unique(y)
    centroids = []
    for label in unique_labels:
        # 计算该类别所有样本的平均光谱
        centroids.append(np.mean(X[y == label], axis=0))
    centroids = np.array(centroids)

    # 2. 简易质心分类器预测 (基于余弦相似度)
    # 计算所有样本与所有质心的相似度矩阵
    sim_matrix = cosine_similarity(X, centroids)
    # 取相似度最高的质心作为预测类别
    y_pred = unique_labels[np.argmax(sim_matrix, axis=1)]

    # 3. 计算混淆矩阵
    cm = confusion_matrix(y, y_pred)
    # 归一化 (百分比表示)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 4. 绘图并保存 (这里定义了 plot_path)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Material Separability Confusion Matrix (After Cleaning)')
    plt.ylabel('True Material')
    plt.xlabel('Predicted (Nearest Centroid)')

    plot_path = os.path.join(output_dir, "cleaning_confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()

    # 5. 保存详细文本报告 (您提供的片段部分)
    report = classification_report(y, y_pred, target_names=label_names)
    with open(os.path.join(output_dir, "cleaning_report.txt"), "w", encoding='utf-8') as f:
        f.write("=== Data Cleaning & Separability Report ===\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report)

    print(f"✅ 混淆矩阵已保存至: {plot_path}")
    print(f"✅ 文本报告已保存至: cleaning_report.txt")

# ================= 🚀 3. 主处理流程 =================

def process_and_save_data():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    white_ref, dark_ref = load_calibration_data(WHITE_REF_PATH, DARK_REF_PATH)
    raw_data_buffer = {}

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

            base = os.path.splitext(os.path.basename(hdr_path))[0]
            json_path = os.path.join(ds_config.get("json_dir", ""), base + ".json") if ds_config.get(
                "json_dir") else None
            mask = get_mask_combined(json_path, img_data)
            valid_pixels = img_data[mask]

            if len(valid_pixels) > SAMPLES_PER_IMAGE:
                valid_pixels = valid_pixels[np.random.choice(len(valid_pixels), SAMPLES_PER_IMAGE, replace=False)]
            if len(valid_pixels) > 0: raw_data_buffer[label_id].append(valid_pixels)

    print(f"\n🧹 [阶段2] 数据清洗与特征工程...")
    final_X, final_y, names = [], [], []
    for label_id, pixel_list in raw_data_buffer.items():
        if not pixel_list: continue
        all_pixels = np.vstack(pixel_list)
        label_name = [d['name'] for d in DATASETS if d['label_id'] == label_id][0]
        snv_pixels = apply_snv(all_pixels)
        clean_snv_pixels = filter_impurities(snv_pixels, label_name, threshold=PURITY_THRESHOLD)
        if len(clean_snv_pixels) == 0: continue

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
    # 调用生成报告函数
    generate_cleaning_report(X, y, names, OUTPUT_DIR)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    print(f"💾 数据已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_and_save_data()