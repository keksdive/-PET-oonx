import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import time
import datetime
import tf2onnx

# ================= 1. 硬件检查 =================
gpus = tf.config.list_physical_devices('GPU')
print(f"\n{'=' * 40}")
if len(gpus) > 0:
    print(f"✅ 显卡就绪: {gpus[0].name}")
    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy('mixed_float16')
        print("⚡ 已启用混合精度 (Mixed Precision) 加速")
    except:
        pass
else:
    print("⚠️ 未检测到 GPU，将在 CPU 上运行 Transformer (速度较慢)")
print(f"{'=' * 40}\n")

# ================= 🔧 路径配置 =================
DATA_DIR = r"E:\SPEDATA\NP_newdata"
MODEL_SAVE_DIR = r"D:\DRL\DRL1\models"
if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)

# ⚡ 参数配置
BATCH_SIZE = 2048
EPOCHS = 100


# ================= 2. 数据管道 =================
def create_dataset(X, y, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# ================= 3. Transformer 模型定义 (强制使用) =================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # 1. Normalization & Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # 2. Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_transformer_model(input_shape):
    """
    构建 Hybrid CNN-Transformer 模型
    结合 CNN 的局部特征提取能力和 Transformer 的全局序列建模能力
    """
    inputs = layers.Input(shape=input_shape)

    # 增加一个维度以适配 Conv1D: (Batch, Bands) -> (Batch, Bands, 1)
    x = layers.Reshape((input_shape[0], 1))(inputs)

    # --- Feature Extraction (CNN) ---
    # 先用 CNN 提取波谱的局部特征（波峰/波谷的斜率等）
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)

    # --- Sequence Modeling (Transformer) ---
    # 强制使用 Transformer Encoder
    # num_heads=2: 关注不同的波段组合模式
    x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)

    # --- Classification Head ---
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # 二分类输出 (Sigmoid): 0=背景, 1=PET
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="PET_Transformer_Model")


# ================= 4. 回调函数 =================
class SmartModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, save_dir):
        super(SmartModelCheckpoint, self).__init__()
        self.save_dir = save_dir
        self.best_acc = -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get('val_accuracy')
        if current_acc is not None and current_acc > self.best_acc:
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            filename = f"pet_transformer_{time_str}_acc_{current_acc:.4f}.h5"
            save_path = os.path.join(self.save_dir, filename)
            self.model.save(save_path)
            print(f"\n💾 [新纪录] 精度: {current_acc:.4f} -> 已保存: {filename}")
            self.best_acc = current_acc


# ================= 5. 主流程 =================
if __name__ == "__main__":
    print("🚀 正在加载新生成的二分类数据集 (X.npy, y.npy)...")
    x_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")

    if not os.path.exists(x_path):
        print(f"❌ 错误：文件不存在 {x_path}")
        exit()

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)

    print(f"📊 数据加载完毕: {X.shape}")
    print(f"   正样本(PET): {np.sum(y == 1)} | 负样本(BG/CC/PA): {np.sum(y == 0)}")

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建数据管道
    print("⚡ 构建高速数据流水线...")
    train_ds = create_dataset(X_train, y_train, is_training=True)
    val_ds = create_dataset(X_test, y_test, is_training=False)

    # 构建并编译模型
    model = build_transformer_model(input_shape=(X.shape[1],))
    model.summary()

    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss="binary_crossentropy",  # 适用于 0/1 二分类
                  metrics=["accuracy"])

    print(f"🔥 开始训练 Transformer 模型 (Batch Size={BATCH_SIZE})...")

    # 训练
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[SmartModelCheckpoint(save_dir=MODEL_SAVE_DIR)]
    )

    print("✅ 训练完成，正在导出 ONNX...")

    # 导出最终模型
    final_path = os.path.join(MODEL_SAVE_DIR, "final_transformer_model.h5")
    model.save(final_path)

    # 导出 ONNX (用于C++部署)
    spec = (tf.TensorSpec((None, X.shape[1]), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    onnx_path = os.path.join(MODEL_SAVE_DIR, "pet_transformer.onnx")
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"🏆 ONNX 导出成功: {onnx_path}")