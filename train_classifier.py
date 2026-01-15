import numpy as np
import os
import spectral.io.envi as envi
import cv2
import json
import gc
import tensorflow as tf
from tensorflow.keras import layers, models
import tf2onnx  # 需要 pip install tf2onnx

# ================= 🔧 1. 需要你填写的参数 =================
# 【关键】把 training.py 跑出来的最优波段列表填在这里
# 举例: SELECTED_BANDS = [12, 45, 67, 89, ..., 190]
SELECTED_BANDS = [19, 39, 62, 69, 70, 72, 74, 76, 78, 83, 90, 93, 95, 103, 105, 106, 112, 115, 123, 128, 133, 140, 143, 150, 160, 172, 174, 180, 187, 197]

# 路径设置 (保持不变)
SPE_ROOT = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET"
JSON_ROOT = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\train-PET\fake_images"
WHITE_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\white_ref.hdr"
DARK_REF_HDR = r"L:\12.12数据集（单排光源）\12.12数据集（单排光源）\DWA\black_ref.hdr"

# 输出模型路径
MODEL_SAVE_PATH = "pet_classifier_model"
ONNX_SAVE_PATH = "pet_classifier.onnx"

# 训练参数
SAMPLE_PIXELS_PER_IMAGE = 500  # 分类训练可以多采点样
MAX_TOTAL_SAMPLES = 500000 # 总样本量也可以大一点
BATCH_SIZE = 256
EPOCHS = 300


# =======================================================

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
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        mask = np.zeros(img_shape, dtype=np.uint8)
        found = False
        for shape in data['shapes']:
            lbl = shape['label'].lower()
            pts = np.array(shape['points'], dtype=np.int32)
            if 'no_pet' in lbl or 'background' in lbl:
                cv2.fillPoly(mask, [pts], 2)  # Label 0 in training
                found = True
            elif 'pet' in lbl:
                cv2.fillPoly(mask, [pts], 1)  # Label 1 in training
                found = True
        return mask if found else None
    except:
        return None


def prepare_classification_data():
    if not SELECTED_BANDS:
        raise ValueError("❌ 请先在代码顶部的 SELECTED_BANDS 中填入 DRL 选出的波段索引！")

    print(f"📥 正在加载数据，仅保留选定的 {len(SELECTED_BANDS)} 个波段...")

    white = load_calib_hdr(WHITE_REF_HDR)
    dark = load_calib_hdr(DARK_REF_HDR)
    denom = (white - dark)
    denom[denom == 0] = 1e-6

    # 预先只截取白/黑参考的对应波段，节省计算
    white = white[:, :, SELECTED_BANDS]
    dark = dark[:, :, SELECTED_BANDS]
    denom = denom[:, :, SELECTED_BANDS]

    X_list, y_list = [], []
    all_files = os.listdir(SPE_ROOT)
    spe_files = [f for f in all_files if f.lower().endswith('.spe')]

    for fname in spe_files:
        if len(X_list) * (SAMPLE_PIXELS_PER_IMAGE // 2) > MAX_TOTAL_SAMPLES: break

        base_name = os.path.splitext(fname)[0]
        spe_path = os.path.join(SPE_ROOT, fname)
        hdr_path = os.path.join(SPE_ROOT, base_name + ".hdr")
        json_path = os.path.join(JSON_ROOT, base_name + ".json")

        if not os.path.exists(hdr_path) or not os.path.exists(json_path): continue

        try:
            fix_header_byte_order(hdr_path)
            raw = envi.open(hdr_path, spe_path).load()
            if raw.shape[1] == 208: raw = np.transpose(raw, (0, 2, 1))

            # === 关键步骤：只取选定波段 ===
            raw_selected = raw[:, :, SELECTED_BANDS]

            calib = (raw_selected.astype(np.float32) - dark) / denom
            mask = get_mask_from_json(json_path, (calib.shape[0], calib.shape[1]))

            if mask is None: continue

            current_X, current_y = [], []
            for m_val, target in [(1, 1), (2, 0)]:  # 1=PET, 0=Non-PET
                idx = np.where(mask == m_val)
                if len(idx[0]) > 0:
                    size = min(len(idx[0]), SAMPLE_PIXELS_PER_IMAGE // 2)
                    s_idx = np.random.choice(len(idx[0]), size=size, replace=False)
                    current_X.append(calib[idx[0][s_idx], idx[1][s_idx], :])
                    current_y.append(np.full(size, target))

            if current_X:
                X_list.append(np.concatenate(current_X))
                y_list.append(np.concatenate(current_y))
                print(f"  + 已处理: {fname}", end='\r')

            del raw, raw_selected, calib, mask
            gc.collect()

        except Exception as e:
            print(f"❌ 错误 {fname}: {e}")

    return np.concatenate(X_list), np.concatenate(y_list)


def build_model(input_shape):
    """构建一个适合 C++ 部署的轻量级 MLP 模型"""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # 二分类：输出 0~1 概率
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    # 1. 准备数据
    X, y = prepare_classification_data()
    print(f"\n✅ 数据准备完成。样本形: {X.shape}, 标签形: {y.shape}")

    # 2. 构建与训练模型
    print("🚀 开始训练分类器...")
    model = build_model(input_shape=(len(SELECTED_BANDS),))

    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    # 3. 保存为 H5 (Python用)
    model.save(MODEL_SAVE_PATH + ".h5")
    print(f"💾 Keras 模型已保存至 {MODEL_SAVE_PATH}.h5")

    # 4. 导出为 ONNX (C++用)
    print("🔄 正在导出为 ONNX 格式...")
    spec = (tf.TensorSpec((None, len(SELECTED_BANDS)), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    with open(ONNX_SAVE_PATH, "wb") as f:
        f.write(model_proto.SerializeToString())

    print("=" * 50)
    print(f"🏆 部署文件已生成: {ONNX_SAVE_PATH}")
    print(f"C++ 推理时，请只截取以下 {len(SELECTED_BANDS)} 个通道:")
    print(SELECTED_BANDS)
    print("=" * 50)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    main()