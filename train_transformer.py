import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import os
import json
import glob
import random

# ================= 🔧 配置 =================
CONFIG_FILE = "best_bands_config.json"
DATA_ROOT = r"I:\SPEDATA\NP_data"  # 指向 save_data.py 的输出
MODEL_SAVE_PATH = r"D:\DRL\DRL1\final_model.h5"

BATCH_SIZE = 64
EPOCHS = 50
PIXELS_PER_FILE = 500  # 每次从一个文件里取多少个像素参与训练

# 类别映射 (必须与 save_data.py 的文件夹一致)
CLASS_MAP = {"Background": 0, "PET": 1, "CC": 2, "PA": 3, "PP": 4, "OTHER": 5}
NUM_CLASSES = len(CLASS_MAP)


# ================= 🧠 数据生成器 =================
class SpectralDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list, selected_bands, batch_size=64, samples_per_file=500):
        self.file_list = file_list
        self.selected_bands = selected_bands
        self.batch_size = batch_size
        self.samples_per_file = samples_per_file
        self.indexes = np.arange(len(self.file_list))
        self.on_epoch_end()

    def __len__(self):
        # 估算每个 Epoch 的步数
        return int(len(self.file_list) * 2)  # 这里的系数可调

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # 动态生成一个 Batch 的数据
        batch_X, batch_y = [], []

        while len(batch_X) < self.batch_size:
            # 随机选一个文件
            file_idx = np.random.choice(len(self.file_list))
            file_path = self.file_list[file_idx]

            # 1. 确定 Label
            folder_name = os.path.basename(os.path.dirname(file_path))
            label_id = CLASS_MAP.get(folder_name, 5)  # 默认 OTHER

            # 2. 加载数据
            try:
                img = np.load(file_path).astype(np.float32)  # (H, W, Bands)
            except:
                continue

            # 3. 筛选波段
            if self.selected_bands:
                img = img[:, :, self.selected_bands]

            # 4. 简单的阈值掩膜 (区分背景和前景)
            # 假设取中间几个波段的平均值
            intensity = np.mean(img, axis=2)
            thresh = np.max(intensity) * 0.15

            fg_mask = intensity > thresh
            bg_mask = ~fg_mask

            # 5. 采样 (前景 & 背景)
            # 采样前景 (Label = label_id)
            fg_pixels = img[fg_mask]
            if len(fg_pixels) > 0:
                take = min(len(fg_pixels), self.samples_per_file // 2)
                chosen = fg_pixels[np.random.choice(len(fg_pixels), take)]
                batch_X.append(chosen)
                batch_y.append(np.full(take, label_id))

            # 采样背景 (Label = 0)
            bg_pixels = img[bg_mask]
            if len(bg_pixels) > 0:
                take = min(len(bg_pixels), self.samples_per_file // 2)
                chosen = bg_pixels[np.random.choice(len(bg_pixels), take)]
                batch_X.append(chosen)
                batch_y.append(np.full(take, 0))  # 背景 Label 0

        # 整理 Batch
        X_out = np.vstack(batch_X)
        y_out = np.concatenate(batch_y)

        # 截取精确的 Batch Size (或者大一点也可以)
        indices = np.random.choice(len(X_out), self.batch_size)
        return X_out[indices], y_out[indices]


# ================= 🚀 主程序 =================
def build_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Reshape((input_dim, 1)),
        layers.Conv1D(32, 5, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # 1. 加载波段配置
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        bands = config["selected_bands"]
    print(f"🤖 使用波段: {bands}")

    # 2. 扫描所有 .npy 文件
    all_files = glob.glob(os.path.join(DATA_ROOT, "**", "*.npy"), recursive=True)
    random.shuffle(all_files)

    # 划分训练/验证 (按文件划分)
    split = int(len(all_files) * 0.8)
    train_files = all_files[:split]
    val_files = all_files[split:]

    print(f"📂 发现 {len(all_files)} 个文件. Train: {len(train_files)}, Val: {len(val_files)}")

    # 3. 创建生成器
    train_gen = SpectralDataGenerator(train_files, bands, batch_size=BATCH_SIZE)
    val_gen = SpectralDataGenerator(val_files, bands, batch_size=BATCH_SIZE)

    # 4. 训练
    model = build_model(len(bands), NUM_CLASSES)
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True),
            callbacks.EarlyStopping(patience=5)
        ]
    )
    print(f"💾 模型已保存: {MODEL_SAVE_PATH}")