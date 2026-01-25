# hsi_predictor_core.py
import os
import time
import json
import numpy as np
import spectral.io.envi as envi
import cv2

# [Matplotlib 设置]
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

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
    if TF_AVAILABLE:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass


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


# ================= 1. 核心预测引擎 =================
class HSIPredictor:
    def __init__(self, model_path, config_path, white_ref_path, dark_ref_path):
        configure_gpu_memory()

        self.model_path = model_path
        self.config_path = config_path
        self.model_type = "unknown"
        self.tf_model = None
        self.onnx_session = None
        self.model_input_dim = 0

        # 1. 加载配置和模型
        initial_bands = self._load_band_config()
        self._load_model(model_path)

        # 2. 确定“目标波段列表”
        self.target_bands = self._adapt_input_shape(initial_bands)

        # 3. 加载校准文件
        print("📥 Loading calibration files...")
        self.white_ref = self._load_spe_calibration(white_ref_path)
        self.dark_ref = self._load_spe_calibration(dark_ref_path)

        calib_bands = min(len(self.white_ref), len(self.dark_ref))
        print(f"ℹ️ Calibration has {calib_bands} bands. Model expects {self.model_input_dim} bands.")

        # 4. 预热
        print("🔥 Warming up model...")
        dummy = np.zeros((1, self.model_input_dim), dtype=np.float32)
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
        expected = 0
        if self.model_type == "keras":
            expected = self.tf_model.input_shape[-1]
        elif self.model_type == "onnx":
            shape = self.onnx_session.get_inputs()[0].shape
            expected = shape[1] if len(shape) == 2 else len(config_bands)

        self.model_input_dim = expected

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

    def predict_image(self, input_path, brightness_thresh=0.1, high_brightness_thresh=0.95, conf_thresh=0.85):
        """
        核心预测逻辑
        参数含义变更:
        - brightness_thresh: 相对比率 (0.0-1.0)，例如 0.1 代表过滤掉 MaxBrightness * 0.1 以下的像素
        - high_brightness_thresh: 相对比率 (0.0-1.0)，例如 0.95 代表过滤掉 MaxBrightness * 0.95 以上的像素
        """
        t_start = time.time()

        # 1. 读取数据
        hdr, spe = self._resolve_paths(input_path)
        self._fix_header(hdr)

        try:
            raw_img = envi.open(hdr, spe).load()
        except Exception as e:
            return None, None, {"error": str(e)}

        if raw_img.shape[1] > 200 and raw_img.shape[1] < 250 and raw_img.shape[2] != raw_img.shape[1]:
            raw_img = np.transpose(raw_img, (0, 2, 1))

        H, W, B_real = raw_img.shape

        # 智能波段兼容
        max_valid_band = min(B_real, len(self.white_ref), len(self.dark_ref))

        # 2. 准备画布
        heatmap_canvas = np.zeros((H, W), dtype=np.float32)

        # 3. 提取有效反射率
        raw_valid = raw_img[:, :, :max_valid_band]
        white_valid = self.white_ref[:max_valid_band]
        dark_valid = self.dark_ref[:max_valid_band]

        diff = (white_valid - dark_valid)
        diff[diff == 0] = 1e-6
        reflectance = (raw_valid - dark_valid) / diff

        # 4. [算法升级] 全相对亮度过滤 (Relative Brightness Filter)
        mean_intensity = np.mean(reflectance, axis=2)

        # 获取当前图像的最大亮度 (基准)
        image_max_val = np.max(mean_intensity) if mean_intensity.size > 0 else 1.0
        if image_max_val == 0: image_max_val = 1.0  # 防止全黑

        # 下限: < Max * 10% (去除背景)
        dynamic_min = image_max_val * brightness_thresh

        # 上限: > Max * 95% (去除高光)
        dynamic_max = image_max_val * high_brightness_thresh

        # 掩膜逻辑: 介于两者之间
        valid_mask = (mean_intensity > dynamic_min) & (mean_intensity < dynamic_max)

        pet_pixels = 0
        inf_time = 0

        if np.sum(valid_mask) > 0:
            valid_pixels = reflectance[valid_mask]

            # 5. 归一化
            p_min = np.min(valid_pixels, axis=1, keepdims=True)
            p_max = np.max(valid_pixels, axis=1, keepdims=True)
            denom = p_max - p_min
            denom[denom < 1e-6] = 1.0
            valid_pixels_norm = (valid_pixels - p_min) / denom

            # 6. 对齐模型输入
            req_dim = self.model_input_dim
            if len(self.target_bands) >= req_dim:
                curr_dim = valid_pixels_norm.shape[1]
                if curr_dim >= req_dim:
                    model_input = valid_pixels_norm[:, :req_dim]
                else:
                    padding = np.zeros((valid_pixels_norm.shape[0], req_dim - curr_dim), dtype=np.float32)
                    model_input = np.hstack((valid_pixels_norm, padding))
            else:
                selected_cols = []
                for b_idx in self.target_bands:
                    if b_idx < max_valid_band:
                        selected_cols.append(valid_pixels_norm[:, b_idx:b_idx + 1])
                    else:
                        selected_cols.append(np.zeros((valid_pixels_norm.shape[0], 1), dtype=np.float32))
                model_input = np.hstack(selected_cols)

            # 7. 推理
            t0 = time.time()
            preds = self._internal_predict(model_input)
            inf_time = time.time() - t0

            prob_pet = 1.0 - preds

            # 硬阈值二值化
            final_decision = np.where(prob_pet > conf_thresh, 1.0, 0.0).flatten()

            pet_pixels = np.sum(final_decision > 0)
            heatmap_canvas[valid_mask] = final_decision

        # 8. 可视化
        fig = plt.figure(figsize=(10, 5), dpi=100)

        ax1 = plt.subplot(1, 2, 1)
        show_band = 100 if B_real > 100 else B_real // 2
        ax1.imshow(raw_img[:, :, show_band], cmap='gray')
        ax1.set_title("Raw")
        ax1.axis('off')

        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(heatmap_canvas, cmap='jet', vmin=0, vmax=1)
        ax2.set_title("AI Result")
        ax2.axis('off')

        plt.tight_layout()

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        vis_img_rgba = np.asarray(buf)
        vis_img_rgb = vis_img_rgba[:, :, :3]
        plt.close(fig)

        info = {
            "inf_time": inf_time,
            "total_time": time.time() - t_start,
            "filename": os.path.basename(input_path),
            "pet_pixels": int(pet_pixels),
            "model_type": self.model_type
        }

        return None, vis_img_rgb, info


if __name__ == "__main__":
    pass