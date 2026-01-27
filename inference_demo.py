import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import os
import time
import glob
import json
import cv2
from scipy.signal import savgol_filter
import tensorflow as tf
from tensorflow.keras import layers

# ================= 🔧 配置区域 =================
# [1] 模型路径
MODEL_PATH = r"J:\多酚类\final_cascade_model\cascade_model.h5"

# [2] 配置文件
CONFIG_PATH = r"J:\多酚类\json-procession-result\material_specific_features.json"

# [3] 输入/输出路径
INPUT_DIR = r"E:\SPEDATA\高谱相机数据集\测试集\PET"
OUTPUT_DIR = r"D:\Processed_Result\inference_overlay\123456"  # 结果保存至新文件夹

# [4] 校准文件
WHITE_REF = r"E:\SPEDATA\高谱相机数据集\DWA\white_ref.spe"
DARK_REF = r"E:\SPEDATA\高谱相机数据集\DWA\dark_ref.spe"

# [5] 阈值与显示参数
BRIGHTNESS_THRESHOLD = 0.15  # 亮度下限 (背景)
MAX_BRIGHTNESS_THRESHOLD = 2.00  # 亮度上限 (高光)
# 修改为 0.0，这样所有结果都会被统计，且无需修改主循环代码也能实现全显
CONFIDENCE_THRESHOLDS = {
    "PET": 0.0,
    "PA":  0.0,
    "CC":  0.0
}
ORIGINAL_BANDS = 208
OVERLAY_ALPHA = 0.90  # [新] 预测颜色层的透明度 (0.0~1.0)，越小越透，纹理越明显


# ================= 🧬 自定义层定义 (保持不变) =================

class SpectralAugment(layers.Layer):
    def __init__(self, shift_range=5, scale_range=0.3, noise_std=0.05, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift_range;
        self.scale = scale_range;
        self.noise_std = noise_std

    def call(self, inputs, training=True): return inputs

    def get_config(self):
        config = super().get_config();
        config.update({"shift_range": self.shift, "scale_range": self.scale, "noise_std": self.noise_std});
        return config


class CascadeLogicLayer(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)

    def call(self, inputs):
        pet_prob, pa_prob = inputs
        is_pet = tf.greater(pet_prob, 0.5);
        is_pa = tf.greater(pa_prob, 0.5)
        return tf.where(is_pet, 1.0, tf.where(is_pa, 2.0, 3.0))


class PhysicsAttention(layers.Layer):
    def __init__(self, init_weights=None, **kwargs):
        super().__init__(**kwargs);
        self.init_w = init_weights

    def build(self, input_shape):
        self.phy_w = tf.constant(self.init_w, dtype=tf.float32) if self.init_w is not None else tf.ones(input_shape[-1],
                                                                                                        dtype=tf.float32)
        self.scale = self.add_weight(name='atten_scale', shape=(1,), initializer='ones', trainable=True)

    def call(self, inputs): return inputs

    def get_config(self): return super().get_config()


# ================= 🛠️ 核心工具类 =================

class ModelWrapper:
    def __init__(self, model_path):
        self.type = "keras" if model_path.endswith(".h5") else "onnx"
        print(f"🔌 加载模型 ({self.type}): {os.path.basename(model_path)}")
        if self.type == "keras":
            self.model = tf.keras.models.load_model(model_path, compile=False,
                                                    custom_objects={"SpectralAugment": SpectralAugment,
                                                                    "CascadeLogicLayer": CascadeLogicLayer,
                                                                    "PhysicsAttention": PhysicsAttention})
        else:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.input_name = self.sess.get_inputs()[0].name;
            self.output_names = [o.name for o in self.sess.get_outputs()]

    def predict(self, X):
        if len(X) == 0: return np.array([]), np.array([])  # 处理空输入
        if self.type == "keras":
            preds = self.model.predict(X, verbose=0, batch_size=2048)
            return preds[0], preds[1]
        else:
            preds = self.sess.run(self.output_names, {self.input_name: X})
            return preds[0], preds[1]


# ================= 🧪 预处理算法 =================

def apply_snv(spectra):
    stds = np.std(spectra, axis=1, keepdims=True);
    stds[stds == 0] = 1e-6
    return (spectra - np.mean(spectra, axis=1, keepdims=True)) / stds


def apply_derivative(spectra, window=11, poly=3):
    return savgol_filter(spectra, window_length=window, polyorder=poly, deriv=1, axis=1)


def load_calibration_data(white_path, dark_path):
    def _load(p):
        hdr = os.path.splitext(p)[0] + ".hdr"
        if not os.path.exists(hdr): raise FileNotFoundError(f"Missing HDR: {hdr}")
        return np.mean(envi.open(hdr, p).load(), axis=(0, 1)).astype(np.float32)

    return _load(white_path), _load(dark_path)


def get_selected_bands_indices(config_path):
    if not os.path.exists(config_path): return list(range(416))
    with open(config_path, 'r') as f:
        data = json.load(f)
    selected = set()
    for mat in data['materials'].values(): selected.update(mat['selected_bands'])
    return sorted(list(selected))


# ================= 🚀 优化的单图推理流程 =================
def process_single_image(fpath, model, white_ref, dark_ref, selected_bands_idx):
    start_t = time.time()

    # 1. 加载 ENVI
    hdr = os.path.splitext(fpath)[0] + ".hdr"
    if not os.path.exists(hdr): return None, "HDR not found"
    try:
        raw_data = envi.open(hdr, fpath).load().astype(np.float32)
        if raw_data.shape[1] < raw_data.shape[2] and raw_data.shape[1] in [206, 208, 224]:
            raw_data = np.transpose(raw_data, (0, 2, 1))
        H, W, B = raw_data.shape
    except Exception as e:
        return None, f"加载失败: {e}"

    # 校准对齐
    if white_ref.shape[0] != B:
        w_aligned = cv2.resize(white_ref.reshape(1, -1), (B, 1), interpolation=cv2.INTER_LINEAR).flatten()
        d_aligned = cv2.resize(dark_ref.reshape(1, -1), (B, 1), interpolation=cv2.INTER_LINEAR).flatten()
    else:
        w_aligned, d_aligned = white_ref, dark_ref

    # 2. 计算反射率
    denom = w_aligned - d_aligned;
    denom[denom == 0] = 1e-6
    reflectance = (raw_data - d_aligned) / denom

    # 模型波段对齐
    if B != ORIGINAL_BANDS:
        flat = reflectance.reshape(-1, B)
        flat = cv2.resize(flat, (ORIGINAL_BANDS, flat.shape[0]), interpolation=cv2.INTER_LINEAR)
        reflectance = flat.reshape(H, W, ORIGINAL_BANDS);
        B = ORIGINAL_BANDS

    # 3. 生成掩膜 (自适应动态阈值)
    # 计算真实光强 (归一化到 0~1)
    abs_intensity = np.mean(reflectance, axis=2)
    v_min = np.nanmin(abs_intensity)
    v_max = np.nanmax(abs_intensity)

    if v_max - v_min < 1e-6:
        intensity_relative = np.zeros_like(abs_intensity)
    else:
        intensity_relative = (abs_intensity - v_min) / (v_max - v_min)

    # 应用阈值
    mask_bg = intensity_relative < BRIGHTNESS_THRESHOLD
    mask_glare = intensity_relative > MAX_BRIGHTNESS_THRESHOLD
    mask_invalid = mask_bg | mask_glare

    # 更新 intensity 用于显示
    intensity = intensity_relative

    # ================= 🚨 [修复点] 提前计算 valid_indices 🚨 =================
    # 必须在这里计算，否则下面的 if 检查会报错 NameError
    valid_indices = np.where(~mask_invalid.flatten())[0]

    # 现在可以安全地进行检查了
    if len(valid_indices) / (H * W) > 0.95:
        pass  # print("⚠️ 警告: 几乎全图都被识别为物体，可能背景阈值太低")
    elif len(valid_indices) / (H * W) < 0.001:
        pass  # print("⚠️ 警告: 几乎全图都被过滤了，可能阈值太高")
    # ====================================================================

    # 4. 仅处理有效像素 (Data Initialization with NaN)
    prob_pet_map = np.full((H * W), np.nan, dtype=np.float32)
    prob_pa_map = np.full((H * W), np.nan, dtype=np.float32)

    if len(valid_indices) > 0:
        # 提取有效像素进行预处理
        X_valid = reflectance.reshape(-1, B)[valid_indices]

        # 特征工程
        X_snv = apply_snv(X_valid)
        X_deriv = apply_derivative(X_snv)
        X_full = np.concatenate([X_snv, X_deriv], axis=1)

        try:
            X_input = X_full[:, selected_bands_idx]
        except IndexError:
            return None, "特征索引越界"

        # 5. 模型推理
        t_inf_start = time.time()
        p_pet, p_pa = model.predict(X_input)
        t_inf_end = time.time()

        # 6. 填回全图矩阵
        prob_pet_map[valid_indices] = p_pet.flatten()
        prob_pa_map[valid_indices] = p_pa.flatten()

        inf_time = t_inf_end - t_inf_start
    else:
        inf_time = 0

    # Reshape 回 2D
    prob_pet_map = prob_pet_map.reshape(H, W)
    prob_pa_map = prob_pa_map.reshape(H, W)

    # 计算 CC (排除法)
    prob_cc_map = np.full((H, W), np.nan, dtype=np.float32)
    valid_mask_2d = ~mask_invalid

    if np.any(valid_mask_2d):
        p_pet_valid = prob_pet_map[valid_mask_2d]
        p_pa_valid = prob_pa_map[valid_mask_2d]
        prob_cc_map[valid_mask_2d] = (1.0 - p_pet_valid) * (1.0 - p_pa_valid)

    # 7. 统计
    stats = {
        "PET": np.sum(prob_pet_map[valid_mask_2d] > CONFIDENCE_THRESHOLDS["PET"]),
        "PA": np.sum(prob_pa_map[valid_mask_2d] > CONFIDENCE_THRESHOLDS["PA"]),
        "CC": np.sum(prob_cc_map[valid_mask_2d] > CONFIDENCE_THRESHOLDS["CC"])
    }

    return {
        "prob_pet": prob_pet_map,
        "prob_pa": prob_pa_map,
        "prob_cc": prob_cc_map,

        # ✅ [必须添加这一行] 把原始强度图传出来，主程序要用它画左边的图
        "raw_intensity": intensity,

        "mask_invalid": mask_invalid,
        "stats": stats,
        "inf_time": inf_time,
        "total_time": time.time() - start_t
    }, None

# ================= 🏁 主程序 =================

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    print(f"🔧 读取配置: {CONFIG_PATH}")
    sel_bands = get_selected_bands_indices(CONFIG_PATH)

    print("📥 读取黑白板...")
    try:
        w_ref, d_ref = load_calibration_data(WHITE_REF, DARK_REF)
    except Exception as e:
        print(f"❌ 校准失败: {e}"); exit()

    wrapper = ModelWrapper(MODEL_PATH)

    files = glob.glob(os.path.join(INPUT_DIR, "*.spe"))
    if not files:
        files = glob.glob(os.path.join(INPUT_DIR, "*.hdr"))
        files = [f for f in files if "ref" not in os.path.basename(f)]
        files = [f.replace(".hdr", ".spe") for f in files]

    print(f"📂 找到 {len(files)} 个文件待处理")

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"\n🖼️ 处理: {fname} ...")

        res, err = process_single_image(fpath, wrapper, w_ref, d_ref, sel_bands)
        if err: print(f"   ❌ 失败: {err}"); continue

        # === 🎨 [核心修改] 叠加纹理显示 (Overlay) ===

        H, W = res['raw_intensity'].shape
        mask = res['mask_invalid']

        # 1. 准备底层：原始纹理 (Gray -> BGR)
        # 归一化强度图到 0-255
        raw_norm = cv2.normalize(res['raw_intensity'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 转为 3 通道 BGR，作为底图
        img_base = cv2.cvtColor(raw_norm, cv2.COLOR_GRAY2BGR)

        # 2. 准备顶层：AI 预测颜色 (RGB Probability)
        # 将 NaN 替换为 0
        p_pet = np.nan_to_num(res['prob_pet'], 0)
        p_pa = np.nan_to_num(res['prob_pa'], 0)
        p_cc = np.nan_to_num(res['prob_cc'], 0)

        # ================= 🧹 [已取消] 应用阈值过滤噪点 =================
        # 注释掉下面三行，即可显示所有概率 > 0 的预测结果（全显模式）
        # p_pet[p_pet < CONFIDENCE_THRESHOLDS["PET"]] = 0
        # p_pa[p_pa   < CONFIDENCE_THRESHOLDS["PA"]]  = 0
        # p_cc[p_cc   < CONFIDENCE_THRESHOLDS["CC"]]  = 0
        # ============================================================
        # ============================================================

        img_color = np.zeros((H, W, 3), dtype=np.uint8)
        img_color[..., 2] = (p_pet * 255).astype(np.uint8)  # R: PET
        img_color[..., 1] = (p_pa * 255).astype(np.uint8)  # G: PA
        img_color[..., 0] = (p_cc * 255).astype(np.uint8)  # B: CC

        # 3. 叠加融合 (Blending)
        # 仅在非背景区域进行融合
        # 公式: Output = Base * (1-alpha) + Color * alpha
        img_overlay = img_base.copy()

        # 提取前景区域
        fg_indices = ~mask

        # 使用 addWeighted 进行融合
        # 注意：addWeighted 是全图操作，为了只处理前景，我们先全图融合，再把背景涂黑
        # 或者使用掩膜操作
        img_blended = cv2.addWeighted(img_base, 1.0 - OVERLAY_ALPHA, img_color, OVERLAY_ALPHA, 0)

        # 4. 背景置黑
        # 将背景区域强制设为纯黑 [0,0,0]
        img_final_right = img_blended
        img_final_right[mask] = [0, 0, 0]

        # 添加图例
        cv2.putText(img_final_right, "Overlay: PET(R) PA(G) CC(B)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # 5. 左侧对比图：原始图像 (带标题)
        img_final_left = img_base.copy()
        # 左侧图背景也设为黑，保持一致性，或者保留噪声看原始情况？
        # 用户说“热力图中背景使用黑色”，通常 Raw 图保留原样比较好对比，但为了美观也可以 Mask
        # 这里我们只 Mask 右侧预测图的背景。左侧保留原貌。
        cv2.putText(img_final_left, "Raw Intensity", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 6. 拼接
        combined_img = np.hstack([img_final_left, img_final_right])

        # 保存
        base_name = os.path.splitext(fname)[0]
        vis_path = os.path.join(OUTPUT_DIR, base_name + "_overlay.png")
        cv2.imwrite(vis_path, combined_img)

        print(f"   ✅ 完成 | 耗时: {res['total_time']:.2f}s (推理 {res['inf_time'] * 1000:.0f}ms)")
        print(f"      统计: PET={res['stats']['PET']}, PA={res['stats']['PA']}, CC={res['stats']['CC']}")
        print(f"      已保存: {vis_path}")

    print("\n🎉 所有任务完成！")