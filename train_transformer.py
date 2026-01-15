import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import json


# 读取自动生成的配置文件
CONFIG_FILE = "best_bands_config.json"

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)
        SELECTED_BANDS = data["selected_bands"]
    print(f"🤖 [Auto] 已从配置文件加载 {len(SELECTED_BANDS)} 个波段")
else:
    # 默认回退（如果没有跑 Step 1）
    print("⚠️ 未找到配置文件，使用默认硬编码波段")
    SELECTED_BANDS = [19, 39, 62, ...]

# ... (后续代码保持不变，确保所有用到 SELECTED_BANDS 的地方都使用这个变量)


# 启用混合精度，提升速度 (针对 NVIDIA GPU)
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')


# ================= 1. 优化后的模型架构 (CNN + Transformer) =================
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

    # --- 新增: 1D-CNN 局部特征提取层 ---
    # 捕捉光谱曲线的局部斜率和波峰特征
    x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)

    # --- Transformer 编码层 ---
    x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)

    # 全局池化，比 Flatten 更鲁棒，减少参数量提高速度
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # 输出层：注意混合精度下，最后的激活建议用 float32
    outputs = layers.Dense(1, activation="sigmoid", dtype='float32')(x)

    return models.Model(inputs, outputs)


if __name__ == "__main__":
    BASE_DIR = r"I:\Hyperspectral Camera Dataset\Processed_Data"

    # 加载数据逻辑...
    # X = np.load(...) , y = np.load(...)

    # --- A. 数据增强：SMOTE (解决类别不平衡) ---
    print("⭐ 正在执行 SMOTE 数据增强...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # --- B. 构建并编译模型 ---
    model = build_spectral_transformer(input_shape=(30,))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),  # 稍微调高学习率配合混合精度
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 早停与学习率衰减
    lr_reducer = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("\n🔥 开始训练优化后的模型...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=512,  # 混合精度可以使用更大的 batch size，速度飞快
        callbacks=[early_stop, lr_reducer]
    )

    model.save(os.path.join(BASE_DIR, "optimized_spectral_model.1.0.h5"))
    print("✅ 优化后的模型已保存。")