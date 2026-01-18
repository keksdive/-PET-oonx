import os
import numpy as np
import spectral.io.envi as envi
import glob
import cv2
import json

# ================= ⚙️ 配置区域 =================
# 原始数据源
DATASETS = [
    {"spe_dir": r"M:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET"},
    {"spe_dir": r"E:\SPEDATA\高谱相机数据集\训练集\no_PET\CC"},
    {"spe_dir": r"E:\SPEDATA\高谱相机数据集\训练集\no_PET\PA"}
]

# 校准文件
WHITE_REF = r"M:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\white_ref.hdr"
DARK_REF = r"M:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\black_ref.hdr"

# 输出根目录 (生成的npy将保存在这里)
OUTPUT_ROOT = r"E:\SPEDATA\NP_data"

# 统一波段数 (必须与 DRL 选出的波段数对应的原始输入一致)
TARGET_BANDS = 208


# ===============================================

def fix_header(hdr_path):
    """修复头文件字节序"""
    try:
        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
            c = f.read()
        if "byte order" not in c.lower():
            with open(hdr_path, 'a') as f: f.write("\nbyte order = 0\n")
    except:
        pass


def load_and_calibrate(hdr_path, white, dark, denom):
    """加载 .spe -> 统一波段 -> 辐射校准"""
    try:
        fix_header(hdr_path)
        base = os.path.splitext(hdr_path)[0]
        spe_path = base + ".spe" if os.path.exists(base + ".spe") else base + ".raw"

        # 加载
        raw = np.array(envi.open(hdr_path, spe_path).load(), dtype=np.float32)

        # 维度修正 (H, W, B)
        if raw.shape[1] < raw.shape[2] and raw.shape[1] in [206, 208]:
            raw = np.transpose(raw, (0, 2, 1))

        # 波段对齐
        H, W, C = raw.shape
        if C != TARGET_BANDS:
            flat = raw.reshape(-1, C)
            flat = cv2.resize(flat, (TARGET_BANDS, H * W), interpolation=cv2.INTER_LINEAR)
            raw = flat.reshape(H, W, TARGET_BANDS)

        # 校准 (Raw - Dark) / (White - Dark)
        calib = (raw - dark) / denom

        # 简单的 SNV 预处理 (可选，建议加上)
        # mean = np.mean(calib, axis=2, keepdims=True)
        # std = np.std(calib, axis=2, keepdims=True)
        # calib = (calib - mean) / (std + 1e-6)

        return calib.astype(np.float16)  # 用半精度节省磁盘空间
    except Exception as e:
        print(f"❌ {os.path.basename(hdr_path)}: {e}")
        return None


def determine_category(path_str):
    """根据路径决定子文件夹名称"""
    u = path_str.upper()
    if "CC" in u or "醋酸" in u: return "CC"
    if "PA" in u: return "PA"
    if "PP" in u: return "PP"
    if "PET" in u and "NO_PET" not in u: return "PET"
    return "OTHER"



def resize_bands(data, target_bands):
    """将数据的波段数调整为 target_bands"""
    H, W, C = data.shape
    if C != target_bands:
        print(f"⚠️ 正在将校准数据从 {C} 波段调整为 {target_bands} 波段...")
        flat = data.reshape(-1, C)
        # cv2.resize dsize 是 (width, height)，对应 (bands, pixels)
        flat = cv2.resize(flat, (target_bands, H * W), interpolation=cv2.INTER_LINEAR)
        return flat.reshape(H, W, target_bands)
    return data




def main():
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)

    # 1. 准备校准数据
    print("📥 加载校准板...")
    fix_header(WHITE_REF)
    fix_header(DARK_REF)

    # 手动查找校准数据文件 (应用之前的修复)
    white_base = os.path.splitext(WHITE_REF)[0]
    white_data = white_base + ".spe" if os.path.exists(white_base + ".spe") else white_base + ".raw"
    dark_base = os.path.splitext(DARK_REF)[0]
    dark_data = dark_base + ".spe" if os.path.exists(dark_base + ".spe") else dark_base + ".raw"

    # 加载
    w = np.array(envi.open(WHITE_REF, white_data).load(), dtype=np.float32)
    d = np.array(envi.open(DARK_REF, dark_data).load(), dtype=np.float32)

    # 维度修正 (如果原本就是 (H, W, C) 则不受影响，主要是处理特殊情况)
    # 注意：这里把 208 改为 w.shape[1]，以防原始数据是 206 导致判定失败
    if w.ndim == 3 and w.shape[1] < w.shape[2]: w = np.transpose(w, (0, 2, 1))
    if d.ndim == 3 and d.shape[1] < d.shape[2]: d = np.transpose(d, (0, 2, 1))

    # === 新增：强制对齐校准板波段到 208 ===
    w = resize_bands(w, TARGET_BANDS)
    d = resize_bands(d, TARGET_BANDS)
    # ===================================

    denom = w - d
    denom[denom == 0] = 1e-6

    # 2. 遍历转换
    count = 0
    for ds in DATASETS:
        files = glob.glob(os.path.join(ds["spe_dir"], "**", "*.hdr"), recursive=True)
        for f in files:
            if "ref" in f.lower(): continue

            cat = determine_category(f)
            save_dir = os.path.join(OUTPUT_ROOT, cat)
            if not os.path.exists(save_dir): os.makedirs(save_dir)

            base_name = os.path.splitext(os.path.basename(f))[0]
            save_path = os.path.join(save_dir, base_name + ".npy")
            if os.path.exists(save_path): continue

            # 传入已经 resize 好的 w, d, denom
            img = load_and_calibrate(f, w, d, denom)
            if img is not None:
                np.save(save_path, img)
                count += 1
                if count % 10 == 0: print(f"✅ 已转换 {count} 个文件 -> {cat}")

    print(f"🏁 转换完成！所有 .npy 已保存在 {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()