# hsi_predictor_core.py
import os
import time
import json
import numpy as np
import spectral.io.envi as envi
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2


# ================= 1. 自定义层定义 =================
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


# ================= 2. 核心预测类 (修复版) =================
class HSIPredictor:
    def __init__(self, model_path, config_path, white_ref_path, dark_ref_path):
        self.model_path = model_path
        self.config_path = config_path

        # 1. 加载配置 (暂存)
        initial_bands = self._load_band_config()

        # 2. 加载模型
        self.model = self._load_model_robustly()

        # 3. [关键修复] 检查维度并自动适配
        self.selected_bands = self._adapt_input_shape(self.model, initial_bands)

        # 4. 加载校准文件
        print("📥 [System] Loading calibration files...")
        self.white_ref = self._load_spe_calibration(white_ref_path)
        self.dark_ref = self._load_spe_calibration(dark_ref_path)

        # 5. 预热模型
        print("🔥 [System] Warming up model...")
        dummy_input = np.zeros((1, len(self.selected_bands)), dtype=np.float32)
        try:
            self.model.predict(dummy_input, verbose=0)
            print("✅ [System] Initialization complete.")
        except Exception as e:
            print(f"❌ [System] Warm-up failed: {e}")

    def _load_band_config(self):
        if not os.path.exists(self.config_path):
            print(f"⚠️ Config not found: {self.config_path}, defaulting to empty.")
            return []
        with open(self.config_path, 'r') as f:
            data = json.load(f)
            bands = data.get("selected_bands", [])
            print(f"🤖 [Config] Loaded {len(bands)} bands from json.")
            return bands

    def _load_model_robustly(self):
        print(f"📥 [Model] Loading model from: {self.model_path}")
        custom_objects = {'transformer_encoder': transformer_encoder}
        try:
            # 方案A: 完整加载
            return tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
        except Exception as e_full:
            print(f"⚠️ [Model] load_model failed ({e_full}), trying load_weights...")
            raise RuntimeError(
                f"❌ Model load failed. Please ensure .h5 contains full model structure.\nError: {e_full}")

    def _adapt_input_shape(self, model, config_bands):
        """
        [自动修复逻辑] 对比 模型期望输入 vs 配置文件输入
        """
        # 获取模型输入层形状 (None, 208) -> 208
        try:
            model_expected_bands = model.input_shape[-1]
        except AttributeError:
            # 某些特殊模型结构可能需要 model.layers[0].input_shape
            model_expected_bands = model.layers[0].input_shape[0][1]

        config_len = len(config_bands)

        if model_expected_bands == config_len:
            print(f"✅ [Check] Perfect match: Model expects {model_expected_bands} bands.")
            return config_bands

        print(f"⚠️ [Mismatch] Model expects {model_expected_bands} bands, but config has {config_len}.")

        if model_expected_bands > config_len:
            print(f"   -> 🔄 Auto-Switch: Using Full Spectrum Mode (0 to {model_expected_bands - 1}).")
            # 自动生成 0~207 的列表
            return list(range(model_expected_bands))
        else:
            raise ValueError(
                f"Model needs fewer bands ({model_expected_bands}) than config ({config_len}). Check model file.")

    def _resolve_paths(self, file_path):
        base = os.path.splitext(file_path)[0]
        hdr = base + ".hdr"
        spe = base + ".spe"
        if not os.path.exists(spe) and os.path.exists(base): spe = base
        return hdr, spe

    def _fix_header_byte_order(self, hdr_path):
        if not os.path.exists(hdr_path): return
        try:
            with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            if not any('byte order' in line.lower() for line in lines):
                with open(hdr_path, 'a') as f: f.write('\nbyte order = 0')
        except:
            pass

    def _load_spe_calibration(self, path):
        hdr, spe = self._resolve_paths(path)
        self._fix_header_byte_order(hdr)
        if not os.path.exists(spe):
            raise FileNotFoundError(f"Calibration file missing: {spe}")
        img = envi.open(hdr, spe).load()
        # 确保也是 Float32
        return np.mean(img, axis=(0, 1)).astype(np.float32)

    def predict_image(self, input_path, brightness_thresh=0.01, conf_thresh=0.85):
        t_start = time.time()

        # 1. 加载
        hdr, spe = self._resolve_paths(input_path)
        self._fix_header_byte_order(hdr)

        try:
            raw_img = envi.open(hdr, spe).load()
        except Exception as e:
            return None, None, {"error": str(e)}

        if raw_img.shape[1] == 208 and raw_img.shape[2] != 208:
            raw_img = np.transpose(raw_img, (0, 2, 1))

        H, W, B = raw_img.shape

        # 2. 预览图
        mid_band = raw_img.shape[2] // 2
        raw_preview = raw_img[:, :, mid_band]
        p2, p98 = np.percentile(raw_preview, (2, 98))
        raw_preview_norm = np.clip((raw_preview - p2) / (p98 - p2 + 1e-6), 0, 1)
        raw_rgb = (raw_preview_norm * 255).astype(np.uint8)
        raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_GRAY2RGB)

        # 3. 校准与波段提取
        diff = (self.white_ref - self.dark_ref).astype(np.float32)
        diff[diff == 0] = 1e-6

        # 这里的 selected_bands 已经被 _adapt_input_shape 修正过了
        # 如果是全波段，这里就会取所有波段
        raw_sel = raw_img[:, :, self.selected_bands].astype(np.float32)
        dark_sel = self.dark_ref[self.selected_bands].astype(np.float32)
        diff_sel = diff[self.selected_bands]

        reflectance = (raw_sel - dark_sel) / diff_sel

        # 4. 亮度 Mask
        mean_intensity = np.mean(reflectance, axis=2)
        dynamic_thresh = max(brightness_thresh, np.max(mean_intensity) * 0.1)
        valid_mask = mean_intensity > dynamic_thresh

        overlay_mask = np.zeros((H, W), dtype=np.uint8)
        inf_time = 0

        if np.sum(valid_mask) > 0:
            valid_pixels = reflectance[valid_mask]

            # 归一化
            p_min = np.min(valid_pixels, axis=1, keepdims=True)
            p_max = np.max(valid_pixels, axis=1, keepdims=True)
            denom = p_max - p_min
            denom[denom < 1e-6] = 1.0
            valid_pixels_norm = (valid_pixels - p_min) / denom

            # 推理
            model_input = valid_pixels_norm.reshape(-1, len(self.selected_bands))
            t0 = time.time()
            preds = self.model.predict(model_input, batch_size=4096, verbose=0)
            inf_time = time.time() - t0

            # 结果处理
            prob_pet = 1.0 - preds
            is_pet = (prob_pet > conf_thresh).flatten()

            final_decision = np.zeros(preds.shape[0], dtype=np.uint8)
            final_decision[is_pet] = 1  # PET
            final_decision[~is_pet] = 2  # 背景材质

            overlay_mask[valid_mask] = final_decision

        # 5. 可视化合成
        result_rgb = raw_rgb.copy()

        # PET = 红色 (0, 0, 255) in BGR
        # 其他材质 = 蓝色 (255, 0, 0)
        # 背景 = 保持原样

        colored_layer = result_rgb.copy()
        colored_layer[overlay_mask == 1] = (0, 0, 255)  # Red
        colored_layer[overlay_mask == 2] = (255, 0, 0)  # Blue

        alpha = 0.4
        mask_bool = overlay_mask > 0
        if np.any(mask_bool):
            result_rgb[mask_bool] = cv2.addWeighted(result_rgb, 1 - alpha, colored_layer, alpha, 0)[mask_bool]

        info = {
            "inf_time": inf_time,
            "total_time": time.time() - t_start,
            "filename": os.path.basename(input_path),
            "pet_pixels": int(np.sum(overlay_mask == 1))
        }

        return raw_rgb, result_rgb, info


# ================= 测试代码 =================
if __name__ == "__main__":
    # 请根据实际路径修改
    MODEL = r"D:\HSP_models\MODELS\classic_20260120-2319_acc_0.9994.h5"
    CONFIG = "best_bands_config.json"
    WHITE = r"E:\SPEDATA\高谱相机数据集\DWA\white_ref.spe"
    DARK = r"E:\SPEDATA\高谱相机数据集\DWA\dark_ref.spe"
    # 请确保此路径存在
    TEST_IMG = r"E:\SPEDATA\高谱相机数据集\train_data_fake\train-PET\2025-12-12-17-36-3-589.spe"

    try:
        if not os.path.exists(TEST_IMG):
            print(f"⚠️ 测试图片不存在: {TEST_IMG}")
        else:
            print("🚀 初始化预测器...")
            predictor = HSIPredictor(MODEL, CONFIG, WHITE, DARK)

            print("🚀 Processing test image...")
            raw, res, info = predictor.predict_image(TEST_IMG)

            if raw is not None:
                print(f"✅ Done! Inference time: {info['inf_time']:.4f}s")
                # 显示结果 (需要 OpenCV 窗口)
                cv2.imshow("Original", raw)
                cv2.imshow("Result", res)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"❌ Error: {info.get('error')}")

    except Exception as e:
        print(f"❌ Error during test: {e}")