import numpy as np
import os
import spectral.io.envi as envi
import cv2
import json
import gc

# ================= 🔧 数据集配置 =================
DATASETS = [
    # 1. PET 文件夹 (包含 PET 标注)
    {
        "spe_dir": r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET",
        "json_dir": r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET\fake_images"
    },
    # 2. 非 PET 文件夹 (包含 PP, CC 等标注)
    {
        "spe_dir": r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-noPET",
        "json_dir": r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-noPET\fake_images"
    }
]

# 公共校准文件
WHITE_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\white_ref.hdr"
DARK_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\black_ref.hdr"
SAVE_DIR = r"I:\Hyperspectral Camera Dataset\Nump_data"

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)


# =================================================

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
    if img.shape[1] == 208: img = np.transpose(img, (0, 2, 1))
    return np.array(img, dtype=np.float32)


def get_mask_from_json(json_path, img_shape):
    """
    智能解析 JSON:
    - Label 1: PET
    - Label 2: PP, PE, CC, No_PET (强负样本)
    - Label 0: 剩余未标注区域 (背景)
    """
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        mask = np.zeros(img_shape, dtype=np.uint8)  # 初始化全 0 (背景)
        found_any = False

        for shape in data['shapes']:
            lbl = shape['label'].lower()
            pts = np.array(shape['points'], dtype=np.int32)

            # === 核心逻辑：根据标签名分类 ===
            if 'pet' in lbl and 'no' not in lbl:
                # 是 PET -> Label 1
                cv2.fillPoly(mask, [pts], 1)
                found_any = True
            else:
                # 其他所有标注 (PP, CC, background, no_pet) -> Label 2
                # 这代表“已知非 PET 材质”
                cv2.fillPoly(mask, [pts], 2)
                found_any = True

        return mask if found_any else None
    except Exception as e:
        print(f"JSON 解析错误: {e}")
        return None


def process_and_save_all():
    print("📦 开始智能处理数据...")

    try:
        white = load_calib_hdr(WHITE_REF_HDR)
        dark = load_calib_hdr(DARK_REF_HDR)
        denom = (white - dark)
        denom[denom == 0] = 1e-6
    except Exception as e:
        print(f"❌ 校准文件加载失败: {e}")
        return

    total_success = 0

    for config in DATASETS:
        spe_dir = config["spe_dir"]
        json_dir = config["json_dir"]

        print(f"\n📂 扫描: {spe_dir}")

        if not os.path.exists(spe_dir):
            print(f"⚠️ 路径不存在: {spe_dir}")
            continue

        files = [f for f in os.listdir(spe_dir) if f.lower().endswith('.spe')]

        for fname in files:
            try:
                base_name = os.path.splitext(fname)[0]
                spe_path = os.path.join(spe_dir, fname)
                hdr_path = os.path.join(spe_dir, base_name + ".hdr")
                json_path = os.path.join(json_dir, base_name + ".json")

                if not os.path.exists(hdr_path) or not os.path.exists(json_path):
                    continue

                # 1. 读取数据
                fix_header_byte_order(hdr_path)
                raw = envi.open(hdr_path, spe_path).load()
                if raw.shape[1] == 208: raw = np.transpose(raw, (0, 2, 1))

                # 2. 校准
                calib = (raw.astype(np.float32) - dark) / denom

                # 3. 获取 Mask (智能识别 Label 1 和 2)
                mask = get_mask_from_json(json_path, (calib.shape[0], calib.shape[1]))

                if mask is not None:
                    # 保存 (文件名加前缀区分来源)
                    prefix = "Data"
                    np.save(os.path.join(SAVE_DIR, f"{prefix}_{base_name}_data.npy"), calib)
                    np.save(os.path.join(SAVE_DIR, f"{prefix}_{base_name}_mask.npy"), mask)

                    total_success += 1
                    print(f"  [√] 已保存: {base_name} (含 Label: {np.unique(mask)})")

                del raw, calib, mask
                gc.collect()

            except Exception as e:
                print(f"  [X] 处理失败 {fname}: {e}")

    print(f"\n✨ 全部完成！共生成 {total_success} 组数据。")
    print("Mask 定义: 0=背景, 1=PET, 2=其他材质(PP/CC等)")


if __name__ == "__main__":
    process_and_save_all()