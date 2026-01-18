import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, mixed_precision
import os
import json
import glob
import random

# ================= 🔧 配置 =================
CONFIG_FILE = "best_bands_config.json"
DATA_ROOT = r"E:\SPEDATA\NP_data"  # 请确认此路径与 save_data.py 一致
MODEL_SAVE_PATH = r"D:\DRL\DRL1\final_model.h5"

BATCH_SIZE = 64
EPOCHS = 50
PIXELS_PER_FILE = 1000  # 增加采样点数以提高覆盖率

# 类别映射 (对应 save_data.py 的文件夹)
# 必须与 main.py 的逻辑保持一致
CLASS_MAP = {"Background": 0, "PET": 1, "CC": 2, "PA": 3, "OTHER": 5}
NUM_CLASSES = len(CLASS_MAP)

# 启用混合精度训练 (如果显卡支持，可大幅加速)
try:
    mixed_precision.set_global_policy('mixed_float16')
except:
    pass


# ================= 🧠 数据生成器 =================
class SpectralDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list, selected_bands, batch_size=64, samples_per_file=500):
        self.file_list = file_list
        self.selected_bands = selected_bands  # 这是一个列表，包含去重后的所有特征波段索引
        self.batch_size = batch_size
        self.samples_per_file = samples_per_file
        self.indexes = np.arange(len(self.file_list))
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.file_list) * 2)

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_X, batch_y = [], []

        # 动态尝试多次填充直到 Batch 满
        attempts = 0
        while len(batch_X) < self.batch_size and attempts < 100:
            attempts += 1
            file_idx = np.random.choice(len(self.file_list))
            file_path = self.file_list[file_idx]

            # 1. 确定 Label
            folder_name = os.path.basename(os.path.dirname(file_path))
            # 模糊匹配文件夹名 (例如 "train-PET" -> "PET")
            label_id = 5  # Default OTHER
            for key, val in CLASS_MAP.items():
                if key in folder_name.upper():
                    label_id = val
                    break
            if label_id == 0:  # 如果路径没匹配上，也默认背景
                pass

            # 2. 加载数据
            try:
                img = np.load(file_path).astype(np.float32)
            except:
                continue

            # 3. 筛选波段 (核心修改点)
            # 使用 DRL 选出的特定波段组合
            if self.selected_bands:
                img = img[:, :, self.selected_bands]

            # 4. 前景背景分离采样
            intensity = np.mean(img, axis=2)
            fg_mask = intensity > 0.05

            # 如果是背景类文件夹，或者本身就是暗像素 -> Label 0
            if label_id == 0:
                target_pixels = img.reshape(-1, img.shape[-1])
                target_label = 0
            else:
                # 如果是材质文件夹，取前景 -> Label ID，取背景 -> Label 0
                fg_pixels = img[fg_mask]
                if len(fg_pixels) > 0:
                    take = min(len(fg_pixels), self.samples_per_file)
                    idx = np.random.choice(len(fg_pixels), take)
                    batch_X.append(fg_pixels[idx])
                    batch_y.append(np.full(take, label_id))
                continue  # 背景部分已隐含在其他文件或通过低阈值处理

            if len(target_pixels) > 0:
                take = min(len(target_pixels), self.samples_per_file)
                idx = np.random.choice(len(target_pixels), take)
                batch_X.append(target_pixels[idx])
                batch_y.append(np.full(take, target_label))

        if len(batch_X) == 0:  # 防止空数据报错
            return np.zeros((self.batch_size, len(self.selected_bands))), np.zeros(self.batch_size)

        X_out = np.vstack(batch_X)
        y_out = np.concatenate(batch_y)

        # 截断或填充
        if len(X_out) > self.batch_size:
            indices = np.random.choice(len(X_out), self.batch_size, replace=False)
            return X_out[indices], y_out[indices]
        else:
            # 数据不足时重复填充
            indices = np.random.choice(len(X_out), self.batch_size, replace=True)
            return X_out[indices], y_out[indices]


# ================= 🚀 模型构建 =================
def build_transformer_model(input_dim, num_classes):
    """
    构建一个增强型分类网络，结合 1D-CNN 提取局部光谱特征 和 Transformer 提取全局相关性
    """
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inputs)
    # CNN + Transformer Encoder 结构..

    # 2. 局部特征提取 (CNN)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # 3. 全局特征提取 (Transformer Encoder Block)
    # Multi-Head Attention
    attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.Add()([x, attention_output])  # Residual
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Feed Forward
    ffn = layers.Dense(128, activation="relu")(x)
    ffn = layers.Dense(128)(ffn)  # 保持维度
    x = layers.Add()([x, ffn])  # Residual
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # 4. 分类头
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)  # 确保输出为 float32

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        # ⚠️ 关键：合并去重后的波段
        bands = config.get("all_unique_bands", [])

    print(f"🤖 使用特征波段总数: {len(bands)}")
    print(f"   (包含 PET、CC、PA 的关键特征并集)")

    # 2. 扫描数据
    all_files = glob.glob(os.path.join(DATA_ROOT, "**", "*.npy"), recursive=True)
    random.shuffle(all_files)

    split = int(len(all_files) * 0.8)
    train_files = all_files[:split]
    val_files = all_files[split:]

    print(f"📂 训练集文件: {len(train_files)} | 验证集文件: {len(val_files)}")

    # 3. 训练
    train_gen = SpectralDataGenerator(train_files, bands, batch_size=BATCH_SIZE)
    val_gen = SpectralDataGenerator(val_files, bands, batch_size=BATCH_SIZE)

    model = build_transformer_model(len(bands), NUM_CLASSES)
    model.summary()

    callbacks_list = [
        callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks_list
    )

    print(f"💾 最终模型已保存: {MODEL_SAVE_PATH}")