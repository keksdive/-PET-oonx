import numpy as np
import os
import spectral.io.envi as envi
import cv2
import json
import gc
import tensorflow as tf  # ✅ 修复：必须导入 tensorflow

# ================= 🚀 路径参数设置 =================
# 确保这些变量名在下方函数调用时保持一致
SPE_ROOT = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET"
JSON_ROOT = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET\fake_images"
WHITE_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\white_ref.hdr"
DARK_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\black_ref.hdr"


# ================================================
def load_raw_calibration(file_path):
    """
    读取高光谱相机的校准文件 (.wcor 或 .dcor)
    假设校准数据是 1D 数组（波段平均值）或与图像宽度一致的 2D 数组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到校准文件: {file_path}")

    # 根据你相机的具体格式读取，通常是 float32 的二进制流
    # 如果你的校准文件是 2048 个 float32 数值（对应波段）：
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        return data
    except Exception as e:
        raise Exception(f"读取校准文件失败: {e}")
def fix_header_byte_order(hdr_path):
    if not os.path.exists(hdr_path): return
    try:
        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        if not any('byte order' in line.lower() for line in lines):
            with open(hdr_path, 'a') as f: f.write('\nbyte order = 0')
    except:
        pass


def load_calib_hdr(hdr_path):
    fix_header_byte_order(hdr_path)
    spe_path = hdr_path.replace('.hdr', '.spe')
    if not os.path.exists(spe_path):
        spe_path = os.path.splitext(hdr_path)[0] + ".spe"
    img = envi.open(hdr_path, spe_path).load()
    if img.shape[1] == 208:
        img = np.transpose(img, (0, 2, 1))
    return np.array(img, dtype=np.float32)

# 在 data_preprocessing.py 中添加 SNV 函数
def apply_snv(spectra):
    """
    Standard Normal Variate (SNV) transformation
    论文建议的预处理方法，消除散射效应
    spectra shape: (n_samples, n_bands)
    """
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    # 避免除以0
    std[std == 0] = 1e-6
    return (spectra - mean) / std


def get_mask_from_json(json_path, img_shape):
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mask = np.zeros(img_shape, dtype=np.uint8)
        found = False
        for shape in data['shapes']:
            lbl = shape['label'].lower()
            pts = np.array(shape['points'], dtype=np.int32)
            if 'no_pet' in lbl or 'background' in lbl:
                cv2.fillPoly(mask, [pts], 2)
                found = True
            elif 'pet' in lbl:
                cv2.fillPoly(mask, [pts], 1)
                found = True
        return mask if found else None
    except:
        return None


def load_and_preprocess_data(data_dir, white_path, dark_path, limit_files=2):
    """验证数据加载是否正常的测试函数"""
    print("🧪 正在启动数据预处理测试...")

    try:
        white = load_calib_hdr(white_path)
        dark = load_calib_hdr(dark_path)
        denom = (white - dark)
        denom[denom == 0] = 1e-6

        all_files = os.listdir(data_dir)
        spe_files = [f for f in all_files if f.lower().endswith('.spe')][:limit_files]

        for fname in spe_files:
            base = os.path.splitext(fname)[0]
            spe_path = os.path.join(data_dir, fname)
            hdr_path = os.path.join(data_dir, base + ".hdr")
            json_path = os.path.join(JSON_ROOT, base + ".json")

            if not os.path.exists(json_path):
                print(f"⚠️ 找不到 JSON: {base}.json")
                continue

            raw = envi.open(hdr_path, spe_path).load()
            if raw.shape[1] == 208: raw = np.transpose(raw, (0, 2, 1))
            calib = (raw.astype(np.float32) - dark) / denom
            mask = get_mask_from_json(json_path, (calib.shape[0], calib.shape[1]))

            if mask is not None:
                print(f"✅ 成功加载文件并生成 Mask: {fname}")
                return calib, mask  # 仅返回第一组用于测试验证

    except Exception as e:
        print(f"❌ 预处理失败: {e}")
    return None, None


if __name__ == "__main__":
    # 配置 GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("🚀 GPU 配置成功")
        except Exception as e:
            print(f"⚠️ GPU 配置报错: {e}")

    # --- 🛠 修复位置：确保变量名与顶部定义完全一致 ---
    s_data, c_data = load_and_preprocess_data(
        SPE_ROOT,
        WHITE_REF_HDR,
        DARK_REF_HDR,
        limit_files=2
    )

    if s_data is not None:
        print(f"\n✨ 测试通过！")
        print(f"光谱数据形状: {s_data.shape}")
        print(f"标签数据形状: {c_data.shape}")
    else:
        print("\n❌ 未能成功提取数据，请检查路径或文件名。")