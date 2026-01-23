import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import os
import time
import glob
import gc
import json
import cv2

# ================= 🔧 配置区域 =================
# [1] 模型路径 (支持 .h5 或 .onnx)
MODEL_PATH = r"D:\Processed_Result\67w-38w\models63w\20260123-1516-0.9999-models.h5"
#MODEL_PATH = r"D:\DRL\DRL1\models\final_pet_model.onnx"  # 也可以切换为 ONNX

# [2] 配置文件
CONFIG_PATH = "best_bands_config.json"

# [3] 路径
INPUT_DIR = r"D:\Train_Data\测试集\PET"
OUTPUT_DIR = r"D:\RESULT\Test_Result\1.231522"

# [4] 校准文件
WHITE_REF = r"D:\Train_Data\DWA\white_ref.spe"
DARK_REF = r"D:\Train_Data\DWA\dark_ref.spe"

# 是否保存可视化结果图片 (True=保存, False=不保存)
SAVE_VISUALIZATION = True

# [5] 参数
BRIGHTNESS_THRESHOLD = 0.01
CONFIDENCE_THRESHOLD = 0.80
INFERENCE_BATCH_SIZE = 8192


# ================= 🧠 模型包装类 (兼容 H5/ONNX) =================
class ModelWrapper:
    def __init__(self, model_path):
        self.model_path = model_path
        self.type = "unknown"
        self.session = None
        self.tf_model = None

        if model_path.endswith(".onnx"):
            self.type = "onnx"
            try:
                import onnxruntime as ort
                # 优先使用 CUDA，如果失败则使用 CPU
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.session = ort.InferenceSession(model_path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                print(f"🚀 [Engine] 已加载 ONNX 模型: {model_path}")
            except ImportError:
                print("❌ 错误: 加载 ONNX 需要安装 onnxruntime 库 (pip install onnxruntime-gpu)")
                exit()
        else:
            self.type = "keras"
            try:
                import tensorflow as tf
                # 加载完整模型 (包含结构)，不需要再 build_model
                # 需要提供自定义层 transformer_encoder，否则会报错
                self.tf_model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'transformer_encoder': transformer_encoder}
                )
                print(f"🚀 [Engine] 已加载 Keras H5 模型: {model_path}")
            except Exception as e:
                print(f"❌ Keras 模型加载失败: {e}")
                exit()

    def predict(self, input_data):
        """
        统一预测接口
        input_data: (Batch, Bands)
        """
        if self.type == "onnx":
            # ONNX 推理
            input_feed = {self.input_name: input_data.astype(np.float32)}
            # session.run 返回一个列表，取第一个输出
            preds = self.session.run(None, input_feed)[0]
            return preds
        elif self.type == "keras":
            # Keras 推理
            return self.tf_model.predict(input_data, batch_size=INFERENCE_BATCH_SIZE, verbose=0)


# ================= 🛠️ 辅助函数定义 =================
# 为了加载 H5，必须定义这个 layer (如果模型里有的话)
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    import tensorflow as tf
    from tensorflow.keras import layers
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


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


def safe_extract_bands(raw_img, bands_indices):
    """
    安全提取波段 (索引钳制)
    """
    H, W, C = raw_img.shape
    safe_indices = [min(b, C - 1) for b in bands_indices]
    return raw_img[:, :, safe_indices], safe_indices


def load_calibration_bands(path, bands_indices):
    """
    只加载特定波段的校准数据，减少内存占用
    """
    hdr, spe = resolve_paths(path)
    fix_header_byte_order(hdr)
    if not os.path.exists(spe): raise FileNotFoundError(f"Missing: {spe}")

    # 加载全谱校准
    img = envi.open(hdr, spe).load()
    mean_spec = np.mean(img, axis=(0, 1)).astype(np.float32)

    # 安全切片
    C = len(mean_spec)
    safe_indices = [min(b, C - 1) for b in bands_indices]

    return mean_spec[safe_indices]


# ================= 🔍 单图处理逻辑 =================
def process_single_image(input_path, engine, white_sel, dark_sel, selected_bands):
    filename = os.path.basename(input_path)
    t_start = time.time()

    # 1. 加载图像
    hdr, spe = resolve_paths(input_path)
    fix_header_byte_order(hdr)
    try:
        raw_img = envi.open(hdr, spe).load()
    except Exception as e:
        return None, f"文件损坏: {e}"

    # 维度修正
    if raw_img.shape[1] > 200 and raw_img.shape[1] < 230 and raw_img.shape[2] != raw_img.shape[1]:
        # 简单判断波段维度是否在第二个位置
        raw_img = np.transpose(raw_img, (0, 2, 1))

    H, W, TotalBands = raw_img.shape

    # 2. 安全提取特征波段 (Raw DN)
    raw_sel, _ = safe_extract_bands(raw_img, selected_bands)
    raw_sel = raw_sel.astype(np.float32)

    # 3. 辐射校准 (只计算这30个波段)
    diff = (white_sel - dark_sel)
    diff[diff == 0] = 1e-6
    reflectance = (raw_sel - dark_sel) / diff

    # 4. 亮度 Mask (基于30个波段的平均亮度)
    mean_intensity = np.mean(reflectance, axis=2)
    dynamic_thresh = max(BRIGHTNESS_THRESHOLD, np.max(mean_intensity) * 0.1)
    valid_mask = mean_intensity > dynamic_thresh

    final_map = np.zeros((H, W), dtype=np.float32)
    inf_time = 0

    if np.sum(valid_mask) > 0:
        valid_pixels = reflectance[valid_mask]

        # 5. Min-Max 归一化 (Pixel-wise)
        p_min = np.min(valid_pixels, axis=1, keepdims=True)
        p_max = np.max(valid_pixels, axis=1, keepdims=True)
        denom = p_max - p_min
        denom[denom < 1e-6] = 1.0

        model_input = (valid_pixels - p_min) / denom

        # 6. AI 推理 (兼容 ONNX/Keras)
        t0 = time.time()
        preds = engine.predict(model_input)
        inf_time = time.time() - t0

        # 7. 结果过滤
        prob_pet = preds  # 假设输出是 PET 概率 (Sigmoid)
        # 如果训练标签是反的 (0=PET)，这里需要 1-preds
        # 根据 save_data.py, PET=1, 所以直接用 preds

        final_decision = np.where(prob_pet > CONFIDENCE_THRESHOLD, 1.0, 0.0)
        final_map[valid_mask] = final_decision.flatten()

    return {
        'map': final_map,
        'raw': raw_img,  # 返回原图用于可视化
        'inf_time': inf_time,
        'total_time': time.time() - t_start
    }, None


# ================= 主程序 =================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 1. 加载波段配置
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            SELECTED_BANDS = json.load(f).get("selected_bands", [])
        print(f"🤖 [Config] 特征波段数: {len(SELECTED_BANDS)}")
    else:
        print("❌ 找不到配置文件！")
        exit()

    # 2. 初始化推理引擎 (自动识别 H5/ONNX)
    engine = ModelWrapper(MODEL_PATH)

    # 3. 预加载校准数据 (只切片出需要的波段)
    print("📥 准备校准数据...")
    try:
        white_sel = load_calibration_bands(WHITE_REF, SELECTED_BANDS)
        dark_sel = load_calibration_bands(DARK_REF, SELECTED_BANDS)
    except Exception as e:
        print(f"❌ 校准文件错误: {e}")
        exit()

    # 4. 批处理
    files = glob.glob(os.path.join(INPUT_DIR, "*.spe"))
    print(f"📂 发现 {len(files)} 个待测文件")

    for fpath in files:
        fname = os.path.basename(fpath)
        res, err = process_single_image(fpath, engine, white_sel, dark_sel, SELECTED_BANDS)

        if err:
            print(f"❌ {fname}: {err}")
        else:
            print(f"✅ {fname} | 推理: {res['inf_time'] * 1000:.1f}ms | 总时: {res['total_time']:.2f}s")

            # 保存结果图
            if SAVE_VISUALIZATION:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1);
                plt.imshow(res['raw'][:, :, 100], cmap='gray');
                plt.title("Raw")
                plt.subplot(1, 2, 2);
                plt.imshow(res['map'], cmap='jet', vmin=0, vmax=1);
                plt.title("AI Result")
                plt.savefig(os.path.join(OUTPUT_DIR, fname + ".png"))
                plt.close()