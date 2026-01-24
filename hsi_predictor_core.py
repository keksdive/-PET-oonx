# hsi_predictor_core.py
import os
import time
import json
import numpy as np
import spectral.io.envi as envi
import cv2

# ================= 0. 环境检测 =================
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not available.")

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime not available.")


def configure_gpu_memory():
    """显存防爆配置"""
    if TF_AVAILABLE:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Keras 模型加载需要的自定义层"""
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


# ================= 1. 核心预测引擎 =================
class HSIPredictor:
    def __init__(self, model_path, config_path, white_ref_path, dark_ref_path):
        configure_gpu_memory()

        self.model_path = model_path
        self.config_path = config_path
        self.model_type = "unknown"
        self.tf_model = None
        self.onnx_session = None
        self.input_name = None

        # 加载配置与模型
        initial_bands = self._load_band_config()
        self._load_model(model_path)
        self.selected_bands = self._adapt_input_shape(initial_bands)

        # 加载校准
        print("📥 Loading calibration files...")
        self.white_ref = self._load_spe_calibration(white_ref_path)
        self.dark_ref = self._load_spe_calibration(dark_ref_path)

        # 预热
        print("🔥 Warming up model...")
        dummy = np.zeros((1, len(self.selected_bands)), dtype=np.float32)
        try:
            self._internal_predict(dummy)
        except Exception as e:
            print(f"❌ Warm-up warning: {e}")

    def _load_band_config(self):
        if not os.path.exists(self.config_path): return []
        with open(self.config_path, 'r') as f:
            return json.load(f).get("selected_bands", [])

    def _load_model(self, path):
        if path.endswith(".onnx"):
            if not ONNX_AVAILABLE: raise ImportError("Need onnxruntime")
            self.model_type = "onnx"
            try:
                self.onnx_session = ort.InferenceSession(path,
                                                         providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            except:
                self.onnx_session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            self.input_name = self.onnx_session.get_inputs()[0].name
        elif path.endswith(".h5"):
            if not TF_AVAILABLE: raise ImportError("Need tensorflow")
            self.model_type = "keras"
            self.tf_model = tf.keras.models.load_model(path,
                                                       custom_objects={'transformer_encoder': transformer_encoder})
        else:
            raise ValueError("Unknown model format")

    def _adapt_input_shape(self, config_bands):
        # 简化版维度适配
        expected = 0
        if self.model_type == "keras":
            expected = self.tf_model.input_shape[-1]
        elif self.model_type == "onnx":
            shape = self.onnx_session.get_inputs()[0].shape
            expected = shape[1] if len(shape) == 2 else len(config_bands)

        if expected != len(config_bands) and isinstance(expected, int):
            print(f"⚠️ Band mismatch: Model {expected} vs Config {len(config_bands)}. Using Model count.")
            return list(range(expected))
        return config_bands

    def _internal_predict(self, input_data):
        if self.model_type == "keras":
            return self.tf_model.predict(input_data, batch_size=4096, verbose=0)
        elif self.model_type == "onnx":
            return self.onnx_session.run(None, {self.input_name: input_data.astype(np.float32)})[0]

    def _resolve_paths(self, file_path):
        base = os.path.splitext(file_path)[0]
        return base + ".hdr", base + ".spe"

    def _fix_header(self, hdr):
        if not os.path.exists(hdr): return
        try:
            with open(hdr, 'r', encoding='utf-8', errors='ignore') as f:
                if 'byte order' not in f.read().lower():
                    with open(hdr, 'a') as fa: fa.write('\nbyte order = 0')
        except:
            pass

    def _load_spe_calibration(self, path):
        hdr, spe = self._resolve_paths(path)
        self._fix_header(hdr)
        if not os.path.exists(spe): raise FileNotFoundError(f"Missing {spe}")
        return np.mean(envi.open(hdr, spe).load(), axis=(0, 1)).astype(np.float32)

    def predict_image(self, input_path, brightness_thresh=0.01, conf_thresh=0.85):
        t_start = time.time()

        # 1. 读取数据
        hdr, spe = self._resolve_paths(input_path)
        self._fix_header(hdr)

        try:
            raw_img = envi.open(hdr, spe).load()
        except Exception as e:
            return None, None, {"error": str(e)}

        # 维度修正 [H, Bands, W] -> [H, W, Bands]
        if raw_img.shape[1] > 200 and raw_img.shape[1] < 250 and raw_img.shape[2] != raw_img.shape[1]:
            raw_img = np.transpose(raw_img, (0, 2, 1))

        H, W, B = raw_img.shape

        # 2. 准备 "画布"
        # 用于生成热力图的灰度底图，默认全 0
        heatmap_canvas = np.zeros((H, W), dtype=np.float32)

        # 3. 校准与切片
        diff = (self.white_ref - self.dark_ref)
        diff[diff == 0] = 1e-6

        raw_sel = raw_img[:, :, self.selected_bands]
        dark_sel = self.dark_ref[self.selected_bands]
        diff_sel = diff[self.selected_bands]

        reflectance = (raw_sel - dark_sel) / diff_sel

        # 4. 亮度过滤
        mean_intensity = np.mean(reflectance, axis=2)
        valid_mask = mean_intensity > brightness_thresh

        pet_pixels = 0
        inf_time = 0

        if np.sum(valid_mask) > 0:
            valid_pixels = reflectance[valid_mask]

            # 归一化
            p_min = np.min(valid_pixels, axis=1, keepdims=True)
            p_max = np.max(valid_pixels, axis=1, keepdims=True)
            denom = p_max - p_min
            denom[denom < 1e-6] = 1.0

            model_input = (valid_pixels - p_min) / denom

            # 推理
            t0 = time.time()
            preds = self._internal_predict(model_input)
            inf_time = time.time() - t0

            # 计算 PET 概率 (假设 0=PET, 1=BG 则需反转; 若直接输出 PET 概率则不需)
            # 根据你之前的逻辑： prob_pet = 1.0 - preds
            prob_pet = 1.0 - preds

            # === [核心修改] 生成纯净热力图 ===

            # 1. 扁平化以便赋值
            probs_flat = prob_pet.flatten()

            # 2. 阈值清洗：低于置信度的直接设为 0 (对应 Jet 里的深蓝)
            # 这样背景噪声就会彻底消失，变成纯色
            probs_flat[probs_flat < conf_thresh] = 0

            # 3. 统计像素
            pet_pixels = np.sum(probs_flat > 0)

            # 4. 赋值回画布
            heatmap_canvas[valid_mask] = probs_flat

        # 5. 生成结果图 (纯粹的 Colormap，不叠加原图)
        # 将 0.0-1.0 映射到 0-255
        heatmap_uint8 = (heatmap_canvas * 255).astype(np.uint8)

        # 应用 JET 颜色映射
        # 0 -> 纯蓝 (背景)
        # 128 -> 绿/黄 (中等置信度)
        # 255 -> 纯红 (高置信度)
        result_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # 转为 RGB 供 GUI 显示
        result_rgb_out = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        # 为了兼容接口，raw_rgb 返回一个空的或者简单的图即可，因为 GUI 已经不显示它了
        # 但为了避免报错，还是生成一下
        raw_rgb_dummy = np.zeros_like(result_rgb_out)

        info = {
            "inf_time": inf_time,
            "total_time": time.time() - t_start,
            "filename": os.path.basename(input_path),
            "pet_pixels": int(pet_pixels),
            "model_type": self.model_type
        }

        return raw_rgb_dummy, result_rgb_out, info


# ================= 测试 =================
if __name__ == "__main__":
    pass