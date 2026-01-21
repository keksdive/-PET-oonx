import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import time
import datetime

# ================= 1. 硬件检查与配置 =================
gpus = tf.config.list_physical_devices('GPU')
print(f"\n{'=' * 40}")
print(f"🖥️ 硬件检测结果: 发现 {len(gpus)} 个 GPU")
if len(gpus) == 0:
    print("⚠️ 警告: 未检测到 GPU！模型将使用 CPU 训练，速度会变慢。")
    print("   -> 已自动切换为 '轻量级 CNN' 模型以适应 CPU。")
    USE_TRANSFORMER = False  # 无显卡时，禁用 Transformer
else:
    print(f"✅ 显卡就绪: {gpus[0].name}")
    print("   -> 将使用 'Transformer + CNN' 混合模型。")
    USE_TRANSFORMER = True  # 有显卡时，使用强力模型

    # 启用混合精度加速 (仅限 GPU)
    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy('mixed_float16')
        print("⚡ 已启用混合精度 (Mixed Precision) 加速")
    except:
        pass
print(f"{'=' * 40}\n")

# ================= 🔧 路径配置 =================
DATA_DIR = r"E:\SPEDATA\NP_newdata"
MODEL_SAVE_DIR = r"D:\DRL\DRL1\models"
if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)

# ⚡ 极速配置
BATCH_SIZE = 2048  # 大批量
EPOCHS = 100


# ================= 2. 数据管道优化 (关键提速点) =================
def create_dataset(X, y, is_training=True):
    """
    使用 tf.data API 构建高性能数据管道
    """
    # 1. 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    # 2. 训练集打乱
    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    # 3. 分批
    dataset = dataset.batch(BATCH_SIZE)

    # 4. 【核心优化】缓存与预取
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# ================= 3. 模型定义 =================
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


def build_model(input_shape, use_transformer=True):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(inputs)

    # 通用 CNN 特征提取层
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)

    if use_transformer:
        # === 显卡模式：Transformer ===
        x = transformer_encoder(x, 64, 2, 128, 0.1)
        x = layers.GlobalAveragePooling1D()(x)
    else:
        # === CPU模式：纯 CNN ===
        x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    name = "Transformer_Model" if use_transformer else "Fast_CNN_Model"
    return models.Model(inputs, outputs, name=name)


# ================= 4. 自定义回调函数 (移到全局范围) =================
class SmartModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, save_dir, min_delta=0.001):
        super(SmartModelCheckpoint, self).__init__()
        self.save_dir = save_dir
        self.min_delta = min_delta
        self.best_acc = -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_acc = logs.get('val_accuracy')

        # 如果当前精度 > (历史最高 + 门槛)
        if current_acc is not None and current_acc > (self.best_acc + self.min_delta):
            # 1. 准备文件名
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            acc_str = f"{current_acc:.4f}"

            # Windows 文件名不建议用— (em dash)，改用标准横杠 -
            filename = f"classic_{time_str}_acc_{acc_str}.h5"
            save_path = os.path.join(self.save_dir, filename)

            # 2. 保存模型
            self.model.save(save_path)
            print(f"\n💾 [新纪录] 精度从 {self.best_acc:.4f} 提升至 {current_acc:.4f}，已保存: {filename}")

            # 3. 更新最高分
            self.best_acc = current_acc


# ================= 5. 主流程 =================
if __name__ == "__main__":
    print("🚀 正在加载数据集 (X.npy, y.npy)...")
    x_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")

    if not os.path.exists(x_path):
        print(f"❌ 错误：文件不存在 {x_path}")
        exit()

    # 强制 float32
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)

    # 标签二值化
    y_binary = np.where(y == 1, 1, 0).astype(np.float32)

    print(f"📊 数据加载完毕: {X.shape}, 正样本率: {np.mean(y_binary):.2%}")

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # 构建高速数据管道
    print("⚡ 构建 tf.data 高速流水线...")
    train_ds = create_dataset(X_train, y_train, is_training=True)
    val_ds = create_dataset(X_test, y_test, is_training=False)

    # 构建模型
    model = build_model(input_shape=(X.shape[1],), use_transformer=USE_TRANSFORMER)
    print(f"🏗️ 模型架构: {model.name}")

    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # 实例化自定义回调
    auto_save_callback = SmartModelCheckpoint(
        save_dir=MODEL_SAVE_DIR,
        min_delta=0.0  # 只要有提升就保存
    )

    print(f"🔥 开始训练 (Batch Size={BATCH_SIZE})...")
    start_time = time.time()

    # 训练 (仅一次)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[auto_save_callback]
    )

    total_time = time.time() - start_time
    print(f"✅ 训练完成！总耗时: {total_time / 60:.2f} 分钟")

    # 导出最终模型
    final_path = os.path.join(MODEL_SAVE_DIR, "final_model.h5")
    model.save(final_path)

    # 导出 ONNX
    import tf2onnx

    spec = (tf.TensorSpec((None, X.shape[1]), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    onnx_path = os.path.join(MODEL_SAVE_DIR, "pet_classifier.onnx")
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"🏆 ONNX 导出完成: {onnx_path}")