import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import json
import glob
import sys

# 启用混合精度，提升速度 (针对 NVIDIA GPU)
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

# ================= 🔧 配置区域 =================
CONFIG_FILE = "best_bands_config.json"
# 预处理数据的存放目录 (请确保 save_data.py 输出到了这里)
NPY_DIR = r"D:\DRL\DRL1\processed_data"
# 模型保存目录
MODEL_SAVE_DIR = r"D:\DRL\DRL1\models"

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# ================= 1. 波段加载逻辑 =================
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)
        SELECTED_BANDS = data["selected_bands"]
    print(f"🤖 [Auto] 已从配置文件加载 {len(SELECTED_BANDS)} 个波段")
else:
    # 默认回退（修复了 ... 语法错误）
    print("⚠️ 未找到配置文件，使用默认硬编码波段")
    SELECTED_BANDS = [19, 39, 62, 69, 70, 72, 74, 76, 78, 83, 90, 93, 95, 103, 105, 106, 112, 115, 123, 128, 133, 140,
                      143, 150, 160, 172, 174, 180, 187, 197]


# ================= 2. 优化后的模型架构 (CNN + Transformer) =================
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


def build_spectral_transformer(input_shape):
    inputs = layers.Input(shape=input_shape)
    # (Batch, 30) -> (Batch, 30, 1)
    x = layers.Reshape((input_shape[0], 1))(inputs)

    # --- 1D-CNN 局部特征提取 ---
    x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)

    # --- Transformer 编码层 ---
    x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)

    # 全局池化
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # 输出层 (二分类: 1=PET, 0=非PET/背景)
    outputs = layers.Dense(1, activation="sigmoid", dtype='float32')(x)

    return models.Model(inputs, outputs)


if __name__ == "__main__":
    # ================= 3. 数据加载与预处理 =================
    print(f"📥 正在加载 .npy 数据 (路径: {NPY_DIR})...")

    X_list = []
    y_list = []

    # 查找所有数据文件
    data_files = glob.glob(os.path.join(NPY_DIR, "*_data.npy"))

    if not data_files:
        print(f"❌ 错误：在 {NPY_DIR} 下未找到 .npy 文件！")
        print("   请先运行 save_data.py 进行数据格式转换。")
        sys.exit(1)

    for d_file in data_files:
        try:
            m_file = d_file.replace("_data.npy", "_mask.npy")
            if not os.path.exists(m_file): continue

            # 加载全波段数据
            data = np.load(d_file)  # (H, W, Total_Bands)
            mask = np.load(m_file)  # (H, W)

            # 只取选定的波段
            data_selected = data[:, :, SELECTED_BANDS]

            # --- SNV 预处理 (Paper Optimization) ---
            h, w, c = data_selected.shape
            flat_data = data_selected.reshape(-1, c)

            mean = np.mean(flat_data, axis=1, keepdims=True)
            std = np.std(flat_data, axis=1, keepdims=True)
            std[std == 0] = 1e-6
            flat_data_snv = (flat_data - mean) / std

            flat_mask = mask.reshape(-1)

            # --- 关键：三类采样策略 ---
            # Label 1: PET (正样本)
            # Label 2: 强负样本 (PP, PE, CC 等)
            # Label 0: 弱负样本 (黑色背景)

            idx_pet = np.where(flat_mask == 1)[0]
            idx_mat = np.where(flat_mask == 2)[0]
            idx_bg = np.where(flat_mask == 0)[0]

            # 采样平衡 (防止某张图背景太多淹没数据)
            # 策略：保证 PET 充足，同时引入足够多的非 PET 材质和一部分背景

            # 1. 取 PET (最多 3000)
            if len(idx_pet) > 3000:
                idx_pet = np.random.choice(idx_pet, 3000, replace=False)

            # 2. 取 非PET材质 (最多 2000) -> 它是强干扰项，要多学
            if len(idx_mat) > 2000:
                idx_mat = np.random.choice(idx_mat, 2000, replace=False)

            # 3. 取 背景 (最多 1000) -> 它是弱干扰项，但也得学一点
            if len(idx_bg) > 1000:
                idx_bg = np.random.choice(idx_bg, 1000, replace=False)

            # 添加到列表
            if len(idx_pet) > 0:
                X_list.append(flat_data_snv[idx_pet])
                y_list.append(np.ones(len(idx_pet)))  # Label 1 -> 1 (PET)

            if len(idx_mat) > 0:
                X_list.append(flat_data_snv[idx_mat])
                y_list.append(np.zeros(len(idx_mat)))  # Label 2 -> 0 (非PET)

            if len(idx_bg) > 0:
                X_list.append(flat_data_snv[idx_bg])
                y_list.append(np.zeros(len(idx_bg)))  # Label 0 -> 0 (背景)

        except Exception as e:
            print(f"⚠️ 跳过文件 {os.path.basename(d_file)}: {e}")

    # 合并数据
    if not X_list:
        raise ValueError("❌ 未提取到任何有效样本！请检查 save_data.py 的输出。")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f"✅ 数据加载完毕: 总样本 {len(y)}")
    print(f"   - 正样本 (PET): {np.sum(y == 1)}")
    print(f"   - 负样本 (背景+杂质): {np.sum(y == 0)}")

    # 检查类别数
    if len(np.unique(y)) < 2:
        print("❌ 致命错误：数据中只包含 1 种类别，无法训练！")
        print("   请确保 processed_data 中既包含 PET 文件，也包含 no_PET 文件。")
        sys.exit(1)

    # ================= 4. 训练流程 =================

    # --- SMOTE 数据增强 ---
    print("⭐ 正在执行 SMOTE 类别平衡...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"   - 平衡后样本数: {len(y_res)}")

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # --- 构建模型 ---
    model = build_spectral_transformer(input_shape=(len(SELECTED_BANDS),))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # --- 回调函数 ---
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    checkpoint = ModelCheckpoint(
        os.path.join(MODEL_SAVE_DIR, "best_model.h5"),
        monitor='val_accuracy',
        save_best_only=True
    )

    print("\n🔥 开始训练模型...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=512,
        callbacks=[early_stop, lr_reducer, checkpoint]
    )

    # ================= 5. 保存与导出 =================
    # 保存最终 Keras 模型
    final_path = os.path.join(MODEL_SAVE_DIR, "final_transformer_model.h5")
    model.save(final_path)
    print(f"💾 模型已保存: {final_path}")

    # 导出 ONNX
    import tf2onnx

    onnx_path = os.path.join(MODEL_SAVE_DIR, "pet_classifier.onnx")
    print(f"🔄 正在导出 ONNX: {onnx_path} ...")

    spec = (tf.TensorSpec((None, len(SELECTED_BANDS)), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print("🏆 部署文件生成完毕！")