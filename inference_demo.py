import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import tensorflow as tf
from tensorflow.keras import mixed_precision
import os
import time
import glob
import gc
import json
from data_preprocessing import load_raw_calibration

# ================= 🚀 核心性能优化配置 =================
try:
    mixed_precision.set_global_policy('mixed_float16')
    print("✅ 已启用 Mixed Precision (混合精度) 加速")
except Exception as e:
    print(f"⚠️ 无法启用混合精度: {e}")

INFERENCE_BATCH_SIZE = 8192

# ================= 📁 路径配置区域 =================
# [修改] 模型路径：指向支持多材质分类的新模型
MODEL_PATH = r"D:\DRL\DRL1\final_model.h5"
CONFIG_FILE = "best_bands_config.json"

INPUT_DIR = r"I:\新建文件夹\高谱相机数据集\测试集\PET"
OUTPUT_DIR = r"I:\Hyperspectral Camera Dataset\Inference_Results"

WHITE_REF = r"I:\Hyperspectral Camera Dataset\B_W\bai1.wcor"
DARK_REF = r"I:\Hyperspectral Camera Dataset\B_W\hei1.dcor"

# [关键修改] 动态加载波段配置
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        conf = json.load(f)
        # 优先读取合并后的特征波段并集
        SELECTED_BANDS = conf.get("all_unique_bands", conf.get("selected_bands", []))
    print(f"🤖 已成功从 JSON 加载 {len(SELECTED_BANDS)} 个特征波段")
else:
    # 备选硬编码 (仅用于应急)
    SELECTED_BANDS = [19, 39, 62, 69, 70, 72, 74, 76, 78, 83]
    print("⚠️ 未找到配置文件，使用默认备选波段")

# 类别定义 (0=背景, 1=PET, 2=CC, 3=PA)
TARGET_PET_LABEL = 1
SAVE_VISUALIZATION = True

# ===========================================

def fix_header_byte_order(hdr_path):
    if not os.path.exists(hdr_path): return
    try:
        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        if not any('byte order' in line.lower() for line in lines):
            with open(hdr_path, 'a') as f:
                f.write('\nbyte order = 0')
    except Exception:
        pass

def resolve_paths(file_path):
    base_path = file_path[:-4] if file_path.lower().endswith(('.spe', '.hdr')) else file_path
    hdr_candidates = [base_path + '.hdr', base_path + '.spe.hdr']
    hdr_path = next((p for p in hdr_candidates if os.path.exists(p)), hdr_candidates[0])
    spe_path = base_path + '.spe'
    if not os.path.exists(spe_path) and os.path.exists(base_path):
        spe_path = base_path
    return hdr_path, spe_path

def process_single_image(input_path, model, white_ref, dark_ref):
    filename = os.path.basename(input_path)
    total_start = time.time()

    hdr_path, spe_path = resolve_paths(input_path)
    if not os.path.exists(hdr_path) or not os.path.exists(spe_path):
        return None, f"文件缺失: {filename}"

    fix_header_byte_order(hdr_path)

    try:
        img_obj = envi.open(hdr_path, spe_path)
        raw_img = img_obj.load()
    except Exception as e:
        return None, f"加载失败: {e}"

    if raw_img.shape[1] == 208 and raw_img.shape[2] != 208:
        raw_img = np.transpose(raw_img, (0, 2, 1))

    H, W, B = raw_img.shape
    diff = (white_ref - dark_ref).astype(np.float32)
    diff[diff == 0] = 1e-6

    # 提取特征波段并校准
    raw_selected = raw_img[:, :, SELECTED_BANDS].astype(np.float32)
    dark_selected = dark_ref[SELECTED_BANDS].astype(np.float32)
    diff_selected = diff[SELECTED_BANDS]
    reduced = (raw_selected - dark_selected) / diff_selected

    # [修改] 展平形状动态适配 SELECTED_BANDS 长度
    flattened = reduced.reshape(-1, len(SELECTED_BANDS))

    inference_start = time.time()
    # 执行多分类预测
    preds = model.predict(flattened, batch_size=INFERENCE_BATCH_SIZE, verbose=0)
    inference_time = time.time() - inference_start

    # 提取 PET 类别 (Label 1) 的概率作为热力图
    # 如果是多分类，preds 形状为 (N, Num_Classes)
    prediction_map = preds[:, TARGET_PET_LABEL].reshape(H, W)
    total_time = time.time() - total_start

    return {
        'map': prediction_map,
        'raw': raw_img,
        'inf_time': inference_time,
        'total_time': total_time,
        'shape': (H, W)
    }, None

if __name__ == "__main__":
    plt.ioff()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"🚀 加载模型: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("🔥 正在预热 GPU...")
        dummy_input = np.zeros((INFERENCE_BATCH_SIZE, len(SELECTED_BANDS)))
        model.predict(dummy_input, verbose=0)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit()

    print("📥 加载校准文件...")
    try:
        white = load_raw_calibration(WHITE_REF)
        dark = load_raw_calibration(DARK_REF)
        # 确保校准板波段对齐
        if len(white) != 208:
             print("⚠️ 校准板维度异常，请检查！")
    except Exception as e:
        print(f"❌ 校准文件错误: {e}")
        exit()

    spe_files = glob.glob(os.path.join(INPUT_DIR, "*.spe"))
    print(f"📂 发现 {len(spe_files)} 个待处理文件")

    for file_path in spe_files:
        fname = os.path.basename(file_path)
        gc.collect()

        result, error = process_single_image(file_path, model, white, dark)

        if error:
            print(f"{fname:<30} | ❌ {error}")
            continue

        inf_time = result['inf_time']
        print(f"{fname:<30} | AI推断: {inf_time:.4f}s | ✅ 完成")

        if SAVE_VISUALIZATION:
            fig = plt.figure(figsize=(10, 5))
            raw_img = result['raw']
            show_band = 100 if raw_img.shape[-1] > 100 else raw_img.shape[-1] // 2

            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(raw_img[:, :, show_band], cmap='gray')
            ax1.set_title("Raw Input")
            ax1.axis('off')

            ax2 = plt.subplot(1, 2, 2)
            # 热力图展示 PET 概率
            im = ax2.imshow(result['map'], cmap='jet', vmin=0, vmax=1)
            ax2.set_title("PET Probability (Red=High)")
            plt.colorbar(im, ax=ax2)
            ax2.axis('off')

            plt.savefig(os.path.join(OUTPUT_DIR, fname + "_res.png"))
            plt.close(fig)

    print(f"🎉 处理完成！结果保存至: {OUTPUT_DIR}")