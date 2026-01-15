import numpy as np
import os
import spectral.io.envi as envi
import cv2
import json
import gc

# ================= 🚀 核心路径设置 (已同步) =================
# 1. 光谱文件所在目录
SPE_ROOT = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET"
# 2. JSON 标注所在目录
JSON_ROOT = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET\fake_images"
# 3. 黑白校准文件
WHITE_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\white_ref.hdr"
DARK_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\black_ref.hdr"
# 4. 保存 .npy 文件的目标目录
SAVE_DIR = r"D:\DRL\DRL1\processed_data"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# =========================================================

def fix_header_byte_order(hdr_path):
    """修正 ENVI 头文件的 byte order 问题"""
    if not os.path.exists(hdr_path): return
    try:
        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        if not any('byte order' in line.lower() for line in lines):
            with open(hdr_path, 'a') as f: f.write('\nbyte order = 0')
    except:
        pass


def load_calib_hdr(hdr_path):
    """加载并预处理校准文件"""
    fix_header_byte_order(hdr_path)
    # 自动定位对应的 .spe 文件
    spe_path = hdr_path.replace('.hdr', '.spe')
    if not os.path.exists(spe_path):
        spe_path = os.path.splitext(hdr_path)[0] + ".spe"

    img = envi.open(hdr_path, spe_path).load()
    # 统一转置为 (H, W, B) 格式
    if img.shape[1] == 208:
        img = np.transpose(img, (0, 2, 1))
    return np.array(img, dtype=np.float32)


def get_mask_from_json(json_path, img_shape):
    """解析 JSON 标注"""
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mask = np.zeros(img_shape, dtype=np.uint8)
        found = False
        for shape in data['shapes']:
            lbl = shape['label'].lower()
            pts = np.array(shape['points'], dtype=np.int32)
            # 兼容标签: 1 为 PET, 2 为 背景/非PET
            if 'no_pet' in lbl or 'background' in lbl:
                cv2.fillPoly(mask, [pts], 2)
                found = True
            elif 'pet' in lbl:
                cv2.fillPoly(mask, [pts], 1)
                found = True
        return mask if found else None
    except:
        return None


def process_and_save_all():
    print("📦 启动批量数据转换 (.spe -> .npy) ...")

    # 加载黑白校准基准
    try:
        white = load_calib_hdr(WHITE_REF_HDR)
        dark = load_calib_hdr(DARK_REF_HDR)
        denom = (white - dark)
        denom[denom == 0] = 1e-6
    except Exception as e:
        print(f"❌ 无法加载校准文件: {e}")
        return

    # 扫描 SPE_ROOT
    all_files = os.listdir(SPE_ROOT)
    spe_files = [f for f in all_files if f.lower().endswith('.spe')]

    success_count = 0
    for fname in spe_files:
        base_name = os.path.splitext(fname)[0]
        spe_path = os.path.join(SPE_ROOT, fname)
        hdr_path = os.path.join(SPE_ROOT, base_name + ".hdr")
        json_path = os.path.join(JSON_ROOT, base_name + ".json")

        # 检查必要文件是否存在
        if not os.path.exists(hdr_path) or not os.path.exists(json_path):
            continue

        try:
            # 1. 读取并转置原始数据
            fix_header_byte_order(hdr_path)
            raw = envi.open(hdr_path, spe_path).load()
            if raw.shape[1] == 208:
                raw = np.transpose(raw, (0, 2, 1))

            # 2. 反射率校准
            calib = (raw.astype(np.float32) - dark) / denom

            # 3. 生成 Mask
            mask = get_mask_from_json(json_path, (calib.shape[0], calib.shape[1]))

            if mask is not None:
                # 4. 保存为 .npy 格式以供快速加载
                save_path_data = os.path.join(SAVE_DIR, f"{base_name}_data.npy")
                save_path_mask = os.path.join(SAVE_DIR, f"{base_name}_mask.npy")

                np.save(save_path_data, calib)
                np.save(save_path_mask, mask)

                success_count += 1
                print(f"  [√] 已处理并保存: {base_name}")

            # 内存管理
            del raw, calib, mask
            gc.collect()

        except Exception as e:
            print(f"  [X] 处理失败 {fname}: {e}")

    print(f"\n✨ 批量任务完成！成功转换 {success_count} 组数据。")
    print(f"输出目录: {SAVE_DIR}")


if __name__ == "__main__":
    process_and_save_all()