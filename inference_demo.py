import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import tensorflow as tf
from tensorflow.keras import mixed_precision
import os
import time
import glob
import gc
from data_preprocessing import load_raw_calibration

# ================= 🚀 核心性能优化配置 =================
# 1. 启用混合精度加速 (必须在加载模型前设置)
try:
    mixed_precision.set_global_policy('mixed_float16')
    print("✅ 已启用 Mixed Precision (混合精度) 加速")
except Exception as e:
    print(f"⚠️ 无法启用混合精度: {e}")

# 2. 增大推理 Batch Size (根据显存调整，建议 16384 或 32768)
INFERENCE_BATCH_SIZE = 8192

# ================= 📁 路径配置区域 =================
# [修改] 模型路径：指向优化后的新模型
MODEL_PATH = r"D:\DRL\DRL1\pet_classifier_model.h5"

# 输入文件夹路径
INPUT_DIR = r"I:\新建文件夹\高谱相机数据集\测试集\PET"

# 结果保存文件夹
OUTPUT_DIR = r"I:\Hyperspectral Camera Dataset\Inference_Results"

# 校准文件路径
WHITE_REF = r"I:\Hyperspectral Camera Dataset\B_W\bai1.wcor"
DARK_REF = r"I:\Hyperspectral Camera Dataset\B_W\hei1.dcor"

# [重要] DQN 选出的 30 个波段 (如果你重新跑了DQN，请更新这里)
SELECTED_BANDS = [19, 39, 62, 69, 70, 72, 74, 76, 78, 83, 90, 93, 95, 103, 105, 106, 112, 115, 123, 128, 133, 140, 143, 150, 160, 172, 174, 180, 187, 197]

# 目标标签定义 (0=PET, 1=Non-PET)
TARGET_PET_LABEL = 0
# 是否生成可视化图 (如果只追求极致速度，可设为 False)
SAVE_VISUALIZATION =False


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
    """
    处理单张图片，返回结果和耗时信息
    """
    filename = os.path.basename(input_path)
    total_start = time.time()

    # 1. 路径解析与头文件修复
    hdr_path, spe_path = resolve_paths(input_path)
    if not os.path.exists(hdr_path) or not os.path.exists(spe_path):
        return None, f"文件缺失: {filename}"

    fix_header_byte_order(hdr_path)

    # 2. 加载与预处理
    try:
        img_obj = envi.open(hdr_path, spe_path)
        raw_img = img_obj.load()
    except Exception as e:
        return None, f"加载失败: {e}"

    # BIL 转置处理
    if raw_img.shape[1] == 208 and raw_img.shape[2] != 208:
        raw_img = np.transpose(raw_img, (0, 2, 1))

    H, W, B = raw_img.shape

    # 快速校准 (利用 broadcasting 避免循环)
    # 注意：这里直接操作 float32 以节省内存转换开销
    diff = (white_ref - dark_ref).astype(np.float32)
    diff[diff == 0] = 1e-6

    # 只提取需要的波段进行校准，减少计算量 (这是提速的关键！)
    # 先切片再计算，比先计算全图再切片快 7倍
    raw_selected = raw_img[:, :, SELECTED_BANDS].astype(np.float32)
    dark_selected = dark_ref[SELECTED_BANDS].astype(np.float32)
    diff_selected = diff[SELECTED_BANDS]

    reduced = (raw_selected - dark_selected) / diff_selected

    # 展平准备输入模型
    flattened = reduced.reshape(-1, 30)

    # 3. AI 推断 (计时核心)
    inference_start = time.time()

    # 使用大 Batch Size 进行预测
    preds = model.predict(flattened, batch_size=INFERENCE_BATCH_SIZE, verbose=0)

    inference_time = time.time() - inference_start

    # 4. 结果整形
    if TARGET_PET_LABEL == 0:
        final_labels = 1.0 - preds
    else:
        final_labels = preds

    prediction_map = final_labels.reshape(H, W)
    total_time = time.time() - total_start

    return {
        'map': prediction_map,
        'raw': raw_img,  # 为了画图还是返回原图
        'inf_time': inference_time,
        'total_time': total_time,
        'shape': (H, W)
    }, None


if __name__ == "__main__":
    # 关闭 Matplotlib 的交互模式，防止内存泄漏
    plt.ioff()

    # 1. 准备工作
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"🚀 加载模型: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        # 预热模型 (Warm-up)：跑一次空数据，避免第一次预测计入初始化时间
        print("🔥 正在预热 GPU...")
        dummy_input = np.zeros((INFERENCE_BATCH_SIZE, 30))
        model.predict(dummy_input, verbose=0)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("建议检查: 1.路径是否正确 2.tensorflow版本是否一致")
        exit()

    print("📥 加载校准文件...")
    try:
        white = load_raw_calibration(WHITE_REF)
        dark = load_raw_calibration(DARK_REF)
    except Exception as e:
        print(f"❌ 校准文件错误: {e}")
        exit()

    # 2. 获取文件列表
    spe_files = glob.glob(os.path.join(INPUT_DIR, "*.spe"))
    if not spe_files:
        print("⚠️ 未找到 .spe 后缀文件，尝试扫描所有文件...")
        all_files = glob.glob(os.path.join(INPUT_DIR, "*"))
        spe_files = [f for f in all_files if not f.endswith('.hdr') and os.path.isfile(f)]

    print(f"📂 发现 {len(spe_files)} 个待处理文件")
    print("-" * 75)
    print(f"{'文件名':<30} | {'AI推断(s)':<10} | {'总耗时(s)':<10} | {'状态'}")
    print("-" * 75)

    # 3. 批量循环
    success_count = 0
    total_inf_time = 0

    for file_path in spe_files:
        fname = os.path.basename(file_path)

        # 显式进行垃圾回收，防止大循环内存累积
        gc.collect()

        result, error = process_single_image(file_path, model, white, dark)

        if error:
            print(f"{fname:<30} | {'-':<10} | {'-':<10} | ❌ {error}")
            continue

        inf_time = result['inf_time']
        tot_time = result['total_time']
        total_inf_time += inf_time
        success_count += 1

        print(f"{fname:<30} | {inf_time:.4f}     | {tot_time:.4f}     | ✅ 完成")

        # 4. 绘图并保存
        if SAVE_VISUALIZATION:
            try:
                fig = plt.figure(figsize=(10, 5))

                # 左图：原始图 (取第100波段或中间波段)
                raw_img = result['raw']
                show_band = 100 if raw_img.shape[-1] > 100 else raw_img.shape[-1] // 2

                ax1 = plt.subplot(1, 2, 1)
                ax1.imshow(raw_img[:, :, show_band], cmap='gray')
                ax1.set_title(f"Raw (Band {show_band})")
                ax1.axis('off')

                # 右图：热力图
                ax2 = plt.subplot(1, 2, 2)
                # 使用 jet colormap, 0=Non-PET(blue), 1=PET(red)
                im = ax2.imshow(result['map'], cmap='jet', vmin=0, vmax=1)
                ax2.set_title(f"AI Detection (Red=PET)")
                plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                ax2.axis('off')

                save_name = os.path.splitext(fname)[0] + "_result.png"
                save_path = os.path.join(OUTPUT_DIR, save_name)

                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                plt.close(fig)  # 彻底关闭图像

            except Exception as e:
                print(f"  -> 保存图片失败: {e}")

    # 4. 总结
    print("-" * 75)
    if success_count > 0:
        avg_time = total_inf_time / success_count
        print(f"🎉 处理完成！成功: {success_count}/{len(spe_files)}")
        print(f"⚡ 平均 AI 推断速度: {avg_time:.4f} 秒/张 (Batch={INFERENCE_BATCH_SIZE})")
        print(f"📂 结果已保存至: {OUTPUT_DIR}")
    else:
        print("❌ 没有成功处理任何图片。")