import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import os
import time
import glob
import gc
import json  # ✅ 新增：用于读取波段配置

# ================= 🚀 核心配置 =================
try:
    mixed_precision.set_global_policy('mixed_float16')
    print("✅ 已启用 Mixed Precision (混合精度)")
except Exception as e:
    print(f"⚠️ 无法启用混合精度: {e}")

# 推理 Batch Size
INFERENCE_BATCH_SIZE = 8192

# ================= 📁 路径配置 =================
# [1] 模型权重路径
MODEL_PATH = r"D:\DRL\DRL1\models\classic_20260121-1356_acc_0.9234.h5"

# [2] 自动加载波段配置 (关键修改)
# 确保这个 json 文件在你的项目目录下，或者改成绝对路径
CONFIG_PATH = "best_bands_config.json"

# [3] 输入输出路径
INPUT_DIR = r"E:\SPEDATA\高谱相机数据集\VAL-noPET"
OUTPUT_DIR = r"D:\RESULT\1.22TEST.1.1"

# [4] 校准文件
WHITE_REF = r"E:\SPEDATA\高谱相机数据集\DWA\white_ref.spe"
DARK_REF = r"E:\SPEDATA\高谱相机数据集\DWA\dark_ref.spe"

# [参数] 亮度阈值 (0.10 ~ 0.15)
BRIGHTNESS_THRESHOLD = 0.01
TARGET_PET_LABEL = 0
SAVE_VISUALIZATION = True
# 2. [新增] 提高置信度阈值 (过滤模棱两可的塑料)
CONFIDENCE_THRESHOLD = 0.4  # 只有概率 > 85% 才认为是 PET


# ================= 🔧 自动加载波段逻辑 =================
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config_data = json.load(f)
        SELECTED_BANDS = config_data.get("selected_bands", [])
    print(f"🤖 [Auto] 已从配置文件加载 {len(SELECTED_BANDS)} 个特征波段")
else:
    print(f"❌ 错误：找不到配置文件 {CONFIG_PATH}")
    print("   -> 请确保 train_transformer.py 运行完毕并生成了该文件")
    print("   -> 或者手动在此处填入波段列表")
    exit()


# ================= 🏗️ 模型架构 (必须与训练一致) =================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(inputs)

    # === 前半部分保持一致 ===
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)

    # === 修改这里：移除 Transformer，换回 CNN ===
    # 原来的 Transformer 代码被注释掉或删除
    # x = transformer_encoder(x, 64, 2, 128, 0.1)

    # 换成训练脚本里对应的 CPU 分支代码：
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs)


# ================= 🛠️ 辅助函数 =================
def fix_header_byte_order(hdr_path):
    if not os.path.exists(hdr_path): return
    try:
        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        if not any('byte order' in line.lower() for line in lines):
            with open(hdr_path, 'a') as f: f.write('\nbyte order = 0')
    except:
        pass


def resolve_paths(file_path):
    base = os.path.splitext(file_path)[0]
    hdr = base + ".hdr"
    spe = base + ".spe"
    if not os.path.exists(spe) and os.path.exists(base): spe = base
    return hdr, spe


def load_spe_calibration(path):
    hdr, spe = resolve_paths(path)
    fix_header_byte_order(hdr)
    if not os.path.exists(spe): raise FileNotFoundError(f"Missing: {spe}")
    img = envi.open(hdr, spe).load()
    return np.mean(img, axis=(0, 1)).astype(np.float32)


# ================= 🔍 核心处理逻辑 =================
# ================= 🔍 核心处理函数 (双重阈值版) =================
def process_single_image(input_path, model, white_ref, dark_ref):
    filename = os.path.basename(input_path)
    t_start = time.time()

    # 1. 解析路径与修复头文件
    hdr, spe = resolve_paths(input_path)
    if not os.path.exists(hdr) or not os.path.exists(spe):
        return None, f"文件缺失"

    fix_header_byte_order(hdr)

    # 2. 加载图像
    try:
        raw_img = envi.open(hdr, spe).load()
    except Exception as e:
        return None, f"加载坏损: {e}"

    if raw_img.shape[1] == 208 and raw_img.shape[2] != 208:
        raw_img = np.transpose(raw_img, (0, 2, 1))

    H, W, B = raw_img.shape

    # 3. 辐射校准 (计算反射率)
    diff = (white_ref - dark_ref).astype(np.float32)
    diff[diff == 0] = 1e-6

    # 提取特征波段
    raw_sel = raw_img[:, :, SELECTED_BANDS].astype(np.float32)
    dark_sel = dark_ref[SELECTED_BANDS].astype(np.float32)
    diff_sel = diff[SELECTED_BANDS]

    reflectance = (raw_sel - dark_sel) / diff_sel

    # 4. [阈值过滤 I] 基于亮度的掩膜
    # 计算平均亮度
    mean_intensity = np.mean(reflectance, axis=2)

    # 动态阈值：必须大于绝对阈值(0.15) 且 大于最大亮度的10% (适应不同曝光)
    dynamic_thresh = max(BRIGHTNESS_THRESHOLD, np.max(mean_intensity) * 0.1)
    valid_mask = mean_intensity > dynamic_thresh

    num_valid = np.sum(valid_mask)

    # 初始化结果图 (默认全黑/0.0)
    final_map = np.zeros((H, W), dtype=np.float32)
    inf_time = 0

    if num_valid > 0:
        # 提取有效像素
        valid_pixels = reflectance[valid_mask]

        # -----------------------------------------------------------
        # 🔥 [关键步骤] Pixel-wise Min-Max 归一化
        # 必须 axis=1，让每个像素独立归一化，消除光照不均匀的影响
        # -----------------------------------------------------------
        p_min = np.min(valid_pixels, axis=1, keepdims=True)
        p_max = np.max(valid_pixels, axis=1, keepdims=True)

        denom = p_max - p_min
        denom[denom < 1e-6] = 1.0  # 防止除以0

        valid_pixels_norm = (valid_pixels - p_min) / denom
        # -----------------------------------------------------------

        # 准备输入
        model_input = valid_pixels_norm.reshape(-1, len(SELECTED_BANDS))

        # AI 推理
        t_inf = time.time()
        preds = model.predict(model_input, batch_size=INFERENCE_BATCH_SIZE, verbose=0)
        inf_time = time.time() - t_inf

        # 5. [阈值过滤 II] 基于置信度的双重过滤
        # 先统一转为 "是PET的概率" (0~1)
        if TARGET_PET_LABEL == 0:
            # 如果训练时 0=PET，那么输出越小越是PET
            # 转换后：prob_pet 越大(接近1)越是PET
            prob_pet = 1.0 - preds
        else:
            prob_pet = preds

        # 🔥 硬卡阈值：
        # 只有概率 > 0.85 的才保留为 1.0 (红)
        # 概率 0.6, 0.7 这种模棱两可的，统统变成 0.0 (蓝/背景)
        final_decision = np.where(prob_pet > CONFIDENCE_THRESHOLD, 1.0, 0.0)

        # 填充回原图
        final_map[valid_mask] = final_decision.flatten()

    return {
        'map': final_map,
        'raw': raw_img,
        'inf_time': inf_time,
        'total_time': time.time() - t_start,
        'shape': (H, W)
    }, None


# ================= 主程序 =================
if __name__ == "__main__":
    plt.ioff()
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 1. 初始化模型
    print(f"🚀 初始化模型 (Input Features={len(SELECTED_BANDS)})...")
    try:
        model = build_model(input_shape=(len(SELECTED_BANDS),))
        print(f"📥 加载权重: {MODEL_PATH}")
        model.load_weights(MODEL_PATH)
        # 预热
        model.predict(np.zeros((1, len(SELECTED_BANDS))), verbose=0)
    except Exception as e:
        print(f"❌ 模型加载错误: {e}")
        print("   -> 可能是波段数不匹配，请检查 json 配置文件和 h5 模型是否对应。")
        exit()

    # 2. 加载校准
    print("📥 加载校准文件...")
    try:
        white = load_spe_calibration(WHITE_REF)
        dark = load_spe_calibration(DARK_REF)
    except Exception as e:
        print(f"❌ 校准错误: {e}")
        exit()

    # 3. 处理文件
    files = glob.glob(os.path.join(INPUT_DIR, "*.spe"))
    if not files: files = glob.glob(os.path.join(INPUT_DIR, "*"))
    files = [f for f in files if not f.endswith('.hdr') and os.path.isfile(f)]

    print(f"📂 待处理: {len(files)} 张")
    print("-" * 60)

    count = 0
    t_total = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        gc.collect()

        res, err = process_single_image(fpath, model, white, dark)

        if err:
            print(f"{fname:<20} | ❌ {err}")
            continue

        count += 1
        t_total += res['inf_time']

        if SAVE_VISUALIZATION:
            try:
                fig = plt.figure(figsize=(8, 4))

                raw = res['raw']
                band_idx = raw.shape[2] // 2
                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(raw[:, :, band_idx], cmap='gray')
                ax1.set_title("Raw Image")
                ax1.axis('off')

                ax2 = plt.subplot(1, 2, 2)
                im = ax2.imshow(res['map'], cmap='jet', vmin=0, vmax=1)
                ax2.set_title("AI Result")
                plt.colorbar(im, ax=ax2)
                ax2.axis('off')

                plt.tight_layout()
                save_p = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + "_result.png")
                plt.savefig(save_p, dpi=120)
                plt.close(fig)

                np.save(os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + "_pred.npy"), res['map'])
                print(f"{fname:<20} | ✅ {res['inf_time']:.3f}s | 已保存")
            except Exception as e:
                print(f"{fname:<20} | ✅ {res['inf_time']:.3f}s | 保存失败: {e}")
        else:
            print(f"{fname:<20} | ✅ {res['inf_time']:.3f}s")

    print("-" * 60)
    if count > 0:
        print(f"平均推理速度: {t_total / count:.4f} 秒/张")