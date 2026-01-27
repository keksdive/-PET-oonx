import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import tf2onnx
import datetime

# ================= 1. 全局配置 =================
# [请修改为您的实际数据路径]
DATA_DIR = r"D:\Processed_Result\material-feature"
JSON_PATH = r"D:\Processed_Result\json-procession-result\material_specific_features.json"

# 输出路径
MODEL_SAVE_DIR = r"D:\Processed_Result\final_cascade_model"
RESULT_DIR = r"D:\Processed_Result\final_cascade_results"

if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)

# 显存配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy('mixed_float16')
        print("⚡ 混合精度加速已开启")
    except:
        pass

# 类别定义
NUM_CLASSES = 4
CLASS_NAMES = ["Background", "PET", "PA", "CC"]  # ID: 0, 1, 2, 3


# ================= 2. 数据集成分核查工具 =================
def print_dataset_composition(y, name="Dataset"):
    """
    打印数据集中各类别的具体数量，确保是混合数据集
    """
    unique, counts = np.unique(y, return_counts=True)
    count_dict = dict(zip(unique, counts))

    print(f"\n📊 [{name}] 成分分析:")
    total = len(y)
    for cls_idx, count in count_dict.items():
        # 映射 ID 到名称
        idx = int(cls_idx)
        cls_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
        percent = (count / total) * 100
        print(f"   - {cls_name:<10}: {count:>6} 样本 ({percent:.2f}%)")

    # 验证是否缺类 (我们只关心 1, 2, 3，背景0已被过滤)
    required = {1, 2, 3}
    present = set(unique.astype(int))
    if not required.issubset(present):
        print(f"   ⚠️ 警告: 该数据集中缺失部分目标类别！现有: {present}")
    else:
        print(f"   ✅ 验证通过: 包含所有目标材质(PET/PA/CC)，是混合数据集。")


# ================= 3. 数据处理 =================

def load_and_preprocess_data():
    print("📥 加载数据...")
    X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.float32)

    # 1. 自动补全导数
    if X.shape[1] == 208:
        print("⚠️ 自动计算导数特征...")
        X_deriv = np.gradient(X, axis=1)
        X = np.concatenate([X, X_deriv], axis=1)

    # 2. 读取波段配置
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            config = json.load(f)
        selected_bands = set()
        for mat in config['materials'].values():
            selected_bands.update(mat['selected_bands'])
        selected_bands = sorted(list(selected_bands))
        X = X[:, selected_bands]
        print(f"🔪 特征切片完成: {X.shape} (使用 {len(selected_bands)} 个波段)")
    else:
        print("⚠️ 未找到 JSON，使用全波段")

    # 3. 过滤背景 (ID 0)
    # 我们的目标是区分 PET, PA, CC。背景由预处理阈值处理。
    valid_mask = y != 0
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"🧹 过滤背景后样本数: {len(y)}")

    # 4. 构造多头标签 (Multi-Head Labels)
    # y_pet: 是PET=1, 其他=0
    # y_pa:  是PA=1,  其他=0
    # CC 对应: y_pet=0 且 y_pa=0
    y_pet = np.where(y == 1, 1.0, 0.0).astype(np.float32)
    y_pa = np.where(y == 2, 1.0, 0.0).astype(np.float32)

    return X, y, y_pet, y_pa


# ================= 4. 模型组件 =================

class SpectralAugment(layers.Layer):
    """强数据增强：抗偏移、抗噪声"""

    def __init__(self, shift_range=5, scale_range=0.3, noise_std=0.05, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift_range
        self.scale = scale_range
        self.noise_std = noise_std

    def call(self, inputs, training=True):
        if not training: return inputs
        batch_size = tf.shape(inputs)[0]
        shift = tf.random.uniform([batch_size], minval=-self.shift, maxval=self.shift + 1, dtype=tf.int32)
        x = tf.map_fn(lambda args: tf.roll(args[0], shift=args[1], axis=0), (inputs, shift),
                      fn_output_signature=inputs.dtype)
        gain = tf.random.uniform([batch_size, 1], minval=1.0 - self.scale, maxval=1.0 + self.scale, dtype=inputs.dtype)
        x = x * gain
        noise = tf.random.normal(tf.shape(x), stddev=self.noise_std, dtype=inputs.dtype)
        return x + noise

    def get_config(self):
        config = super().get_config()
        config.update({"shift_range": self.shift, "scale_range": self.scale, "noise_std": self.noise_std})
        return config


class CascadeLogicLayer(layers.Layer):
    """
    [核心逻辑层] In-Graph Logic
    逻辑：First Check PET -> Then Check PA -> Else CC
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs[0]: pet_prob, inputs[1]: pa_prob
        pet_prob, pa_prob = inputs

        is_pet = tf.greater(pet_prob, 0.5)
        is_pa = tf.greater(pa_prob, 0.5)

        # 输出原始 Label ID: 1.0(PET), 2.0(PA), 3.0(CC)
        final_id = tf.where(is_pet, 1.0, tf.where(is_pa, 2.0, 3.0))
        return final_id


# ================= 5. 模型构建 =================

def build_multi_head_model(input_shape):
    inputs = layers.Input(shape=input_shape, name="spectral_input")

    # 增强
    x = SpectralAugment()(inputs)
    x = layers.Reshape((input_shape[0], 1))(x)

    # 主干 (Backbone)
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)

    attn = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    features = layers.GlobalAveragePooling1D()(x)
    features = layers.Dropout(0.5)(features)

    # 头 A: PET 判断
    x_pet = layers.Dense(32, activation="relu")(features)
    out_pet = layers.Dense(1, activation="sigmoid", name="head_pet")(x_pet)

    # 头 B: PA 判断
    x_pa = layers.Dense(32, activation="relu")(features)
    out_pa = layers.Dense(1, activation="sigmoid", name="head_pa")(x_pa)

    # 逻辑层 (用于推理/ONNX)
    final_id = CascadeLogicLayer(name="final_logic")([out_pet, out_pa])

    return models.Model(inputs=inputs, outputs=[out_pet, out_pa, final_id])


# ================= 6. 验证监控回调 =================

class LogicMetrics(callbacks.Callback):
    def __init__(self, val_data):
        super().__init__()
        self.X_val, self.y_val_true = val_data

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            preds = self.model.predict(self.X_val, verbose=0)
            y_pred_id = preds[2].flatten()  # final_logic 输出

            # 监控 PET 召回率
            pet_mask = (self.y_val_true == 1)
            pet_acc = np.mean(y_pred_id[pet_mask] == 1) if np.sum(pet_mask) > 0 else 0

            # 监控 CC 纯度 (是否被误判为 PA)
            cc_mask = (self.y_val_true == 3)
            cc_as_pa = np.mean(y_pred_id[cc_mask] == 2) if np.sum(cc_mask) > 0 else 0

            print(f"\n🧐 [验证集监控] PET召回率: {pet_acc:.4f} | CC被误判为PA率: {cc_as_pa:.4f} (越低越好)")


# ================= 7. 主流程 =================

if __name__ == "__main__":
    # 1. 准备数据
    X, y_raw, y_pet, y_pa = load_and_preprocess_data()
    indices = np.arange(len(X))

    # ================= 🚨 混合数据集划分 (60/20/20) 🚨 =================
    print(f"\n📦 正在进行混合数据集划分 (Stratified Split)...")

    # 第一次切分：留出 20% 作为最终测试集 (Test)
    X_tv, X_test, y_tv, y_test, idx_tv, idx_test = train_test_split(
        X, y_raw, indices, test_size=0.2, stratify=y_raw, random_state=42
    )

    # 第二次切分：从剩余 80% 中留出 25% 作为验证集 (Val) -> 相当于总体的 20%
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_tv, y_tv, idx_tv, test_size=0.25, stratify=y_tv, random_state=42
    )

    # 同步切分多头标签
    y_pet_train = y_pet[idx_train]
    y_pet_val = y_pet[idx_val]
    y_pet_test = y_pet[idx_test]

    y_pa_train = y_pa[idx_train]
    y_pa_val = y_pa[idx_val]
    y_pa_test = y_pa[idx_test]

    # ================= 🔎 核心：验证集成分核查 🔎 =================
    print_dataset_composition(y_train, "Train Set (训练集)")
    print_dataset_composition(y_val, "Val Set (验证集)")
    print_dataset_composition(y_test, "Test Set (测试集)")
    # ==========================================================

    # 2. 准备训练数据字典
    train_inputs = X_train
    train_outputs = {
        "head_pet": y_pet_train,
        "head_pa": y_pa_train,
        "final_logic": y_train  # 占位，loss weight为0
    }

    val_inputs = X_val
    val_outputs = {
        "head_pet": y_pet_val,
        "head_pa": y_pa_val,
        "final_logic": y_val
    }

    # 3. 构建模型
    model = build_multi_head_model((X.shape[1],))

    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss={
            "head_pet": "binary_crossentropy",
            "head_pa": "binary_crossentropy",
            "final_logic": None
        },
        # 权重策略：PET识别最重要(1.0)，其次是PA(0.5)，CC靠排除
        loss_weights={
            "head_pet": 1.0,
            "head_pa": 0.5,
            "final_logic": 0.0
        },
        metrics={"head_pet": "accuracy", "head_pa": "accuracy"}
    )

    # 4. 训练
    print("🔥 开始训练级联逻辑模型...")
    history = model.fit(
        train_inputs, train_outputs,
        validation_data=(val_inputs, val_outputs),
        epochs=100,
        batch_size=256,
        callbacks=[
            callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            LogicMetrics((val_inputs, y_val))  # 自定义监控
        ]
    )

    # 5. 最终评估 (基于 Test 集)
    print("\n📊 最终评估 (Test Set)...")
    raw_preds = model.predict(X_test)
    final_pred_ids = raw_preds[2].flatten()  # 取逻辑层输出

    print(classification_report(y_test, final_pred_ids, target_names=["PET", "PA", "CC"]))

    cm = confusion_matrix(y_test, final_pred_ids)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["PET", "PA", "CC"], yticklabels=["PET", "PA", "CC"])
    plt.title("Final Cascade Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(RESULT_DIR, "cascade_confusion.png"))

    # 6. 保存双格式
    # A. H5
    h5_path = os.path.join(MODEL_SAVE_DIR, "cascade_model.h5")
    model.save(h5_path)
    print(f"✅ H5 模型已保存: {h5_path}")

    # B. ONNX (直接输出 Label ID)
    onnx_path = os.path.join(MODEL_SAVE_DIR, "cascade_model.onnx")
    spec = (tf.TensorSpec((None, X.shape[1]), tf.float32, name="spectral_input"),)

    try:
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        with open(onnx_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        print(f"✅ ONNX 模型已导出: {onnx_path}")
        print("💡 ONNX 输出: [pet_prob, pa_prob, final_label_id]")
    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")