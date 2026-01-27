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

# ================= 1. 全局配置 =================
# [请修改为您的实际数据路径]
DATA_DIR = r"G:\NP_new_MultiClass_SNV"
JSON_PATH = r"G:\多酚类\json-procession-result\material_specific_features.json"

MODEL_SAVE_DIR = r"G:\多酚类\final_cascade_model"
RESULT_DIR = r"G:\多酚类\json-procession-result"

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


# ================= 2. 物理先验解析与特征工程 =================

def parse_physics_priors(json_path, sorted_selected_bands):
    """
    深度解析 JSON，提取物理先验权重
    返回: prior_weights (Num_Bands, 3) -> [Intensity, Slope, Curvature]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 建立映射: 原始波段ID -> 切片后的索引
    band_map = {original_idx: i for i, original_idx in enumerate(sorted_selected_bands)}
    num_bands = len(sorted_selected_bands)

    # 初始化权重: 默认为 1.0
    priors = np.ones((num_bands, 3), dtype=np.float32)

    print("🧠 [Physics] 正在注入 JSON 物理先验知识...")
    count_hits = 0

    for mat_name, mat_data in data['materials'].items():
        if 'band_analysis' not in mat_data: continue

        for item in mat_data['band_analysis']:
            original_idx = item['index']
            if original_idx not in band_map: continue

            idx = band_map[original_idx]

            # 规则 1: 物理匹配
            if "Hit" in item.get('physical_match', ''):
                priors[idx] *= 3.0
                count_hits += 1

            # 规则 2: 特征类型
            if item.get('type') == 'Derivative':
                priors[idx, 1] *= 2.0
                priors[idx, 2] *= 2.0

                # 规则 3: 拓扑预期
            topo = item.get('topology_expect', 'Unknown')
            if topo in ['Valley', 'Peak', 'LocalMin', 'LocalMax']:
                priors[idx, 2] *= 4.0

    priors = priors / np.mean(priors)
    print(f"   -> 已注入 {count_hits} 个物理特征点，生成先验矩阵 {priors.shape}")
    return priors


def compute_physics_features(X):
    """
    输入: (Batch, Bands)
    输出: (Batch, Bands, 3) -> [Intensity, Slope, Curvature]
    """
    f0 = X
    f1 = np.gradient(f0, axis=1)
    f2 = np.gradient(f1, axis=1)
    return np.stack([f0, f1, f2], axis=-1).astype(np.float32)


# ================= 3. 自定义模型层 (必须在模型构建前定义) =================

class SpectralAugment(layers.Layer):
    """支持 3通道 (Bands, 3) 的数据增强"""

    def __init__(self, shift_range=5, scale_range=0.3, noise_std=0.05, **kwargs):
        super().__init__(**kwargs)
        self.shift = shift_range
        self.scale = scale_range
        self.noise_std = noise_std

    def call(self, inputs, training=True):
        if not training: return inputs
        batch_size = tf.shape(inputs)[0]
        # Shift
        shift = tf.random.uniform([batch_size], minval=-self.shift, maxval=self.shift + 1, dtype=tf.int32)
        x = tf.map_fn(lambda args: tf.roll(args[0], shift=args[1], axis=0), (inputs, shift),
                      fn_output_signature=inputs.dtype)
        # Gain
        gain = tf.random.uniform([batch_size, 1, 1], minval=1.0 - self.scale, maxval=1.0 + self.scale,
                                 dtype=inputs.dtype)
        x = x * gain
        # Noise
        noise = tf.random.normal(tf.shape(x), stddev=self.noise_std, dtype=inputs.dtype)
        return x + noise

    def get_config(self):
        config = super().get_config()
        config.update({"shift_range": self.shift, "scale_range": self.scale, "noise_std": self.noise_std})
        return config


class PhysicsGuidedAttention(layers.Layer):
    """
    [修复版] 兼容混合精度 (Mixed Precision) 的物理注意力层
    """

    def __init__(self, prior_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.init_weights = prior_weights  # numpy array

    def build(self, input_shape):
        # input_shape: (Batch, Bands, 3)
        num_bands = input_shape[1]
        channels = input_shape[2]

        # 1. 可学习的注意力核
        self.attn_kernel = self.add_weight(
            name="attn_kernel",
            shape=(1, 1, channels),
            initializer="glorot_uniform",
            trainable=True
        )

        # 2. 物理先验缩放因子 (保持 float32 以存储高精度先验)
        if self.init_weights is not None:
            w = np.expand_dims(self.init_weights, axis=0)
            self.prior_scale = tf.constant(w, dtype=tf.float32)
        else:
            self.prior_scale = tf.ones((1, num_bands, channels), dtype=tf.float32)

    def call(self, inputs):
        # inputs: [Intensity, Slope, Curvature]
        # 在混合精度下，inputs.dtype 通常是 float16

        # ================= 🚨 [修复关键点] 🚨 =================
        # 将 float32 的先验权重转换为当前计算精度 (float16)
        scale = tf.cast(self.prior_scale, dtype=inputs.dtype)
        # ====================================================

        # 现在 float16 * float16，不会报错了
        weights = tf.sigmoid(self.attn_kernel) * scale

        return inputs * weights

    def get_config(self):
        config = super().get_config()
        if self.init_weights is not None:
            config.update({"prior_weights": self.init_weights.tolist()})
        return config

class CascadeLogicLayer(layers.Layer):
    """PET -> PA -> PC -> Else CC"""

    def __init__(self, **kwargs): super().__init__(**kwargs)

    def call(self, inputs):
        pet_p, pa_p, pc_p = inputs
        is_pet = tf.greater(pet_p, 0.5)
        is_pa = tf.greater(pa_p, 0.5)
        is_pc = tf.greater(pc_p, 0.5)

        # 输出 ID: 1(PET), 2(PA), 4(PC), 3(CC)
        # 注意: 这里假设没有PC数据时 pc_p会很小, 不影响逻辑
        return tf.where(is_pet, 1.0,
                        tf.where(is_pa, 2.0,
                                 tf.where(is_pc, 4.0, 3.0)))


# ================= 4. 数据加载与处理流程 =================

def load_and_process_data():
    print("📥 加载数据...")
    X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.float32)

    # 读取 JSON 波段配置
    selected_bands = list(range(X.shape[1]))  # 默认全波段
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        sb_set = set()
        for mat in config['materials'].values():
            sb_set.update(mat['selected_bands'])
        selected_bands = sorted(list(sb_set))

        # 切片
        X = X[:, selected_bands]
        print(f"🔪 特征切片完成: {len(selected_bands)} bands")

    # 1. 计算先验权重
    prior_matrix = parse_physics_priors(JSON_PATH, selected_bands)

    # 2. 计算多维物理特征
    print("🔨 计算二阶导数等物理特征...")
    X_physics = compute_physics_features(X)

    # 3. 过滤背景 (ID 0)
    valid_mask = y != 0
    X_physics = X_physics[valid_mask]
    y = y[valid_mask]

    print(f"🧹 最终样本数: {len(y)}")

    # 4. 构造多头标签
    # 假设 ID: 1=PET, 2=PA, 3=CC, 4=PC
    y_pet = np.where(y == 1, 1.0, 0.0).astype(np.float32)
    y_pa = np.where(y == 2, 1.0, 0.0).astype(np.float32)
    y_pc = np.where(y == 4, 1.0, 0.0).astype(np.float32)

    return X_physics, y, y_pet, y_pa, y_pc, prior_matrix


# ================= 5. 模型构建函数 =================

def build_physics_model(input_shape, prior_weights):
    # input_shape: (Bands, 3)
    inputs = layers.Input(shape=input_shape, name="physics_input")

    # 1. 数据增强
    x = SpectralAugment()(inputs)

    # 2. 物理注意力 (注入先验) - 现在这个类已经定义了，不会报错
    x = PhysicsGuidedAttention(prior_weights=prior_weights, name="physics_attention")(x)

    # 3. 特征提取 (CNN)
    x = layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)

    # 4. 全局特征
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)

    # 5. 独立决策头
    out_pet = layers.Dense(1, activation='sigmoid', name="head_pet")(layers.Dense(32, activation='relu')(x))
    out_pa = layers.Dense(1, activation='sigmoid', name="head_pa")(layers.Dense(32, activation='relu')(x))
    out_pc = layers.Dense(1, activation='sigmoid', name="head_pc")(layers.Dense(32, activation='relu')(x))

    # 6. 逻辑输出
    final_id = CascadeLogicLayer(name="final_logic")([out_pet, out_pa, out_pc])

    return models.Model(inputs=inputs, outputs=[out_pet, out_pa, out_pc, final_id])


# ================= 6. 主程序 =================

if __name__ == "__main__":
    # 1. 加载数据与计算特征
    X, y_raw, y_pet, y_pa, y_pc, prior_matrix = load_and_process_data()

    # 2. 划分数据集
    indices = np.arange(len(X))
    X_train, X_test, idx_train, idx_test = train_test_split(
        X, indices, test_size=0.2, stratify=y_raw, random_state=42
    )

    train_outputs = {
        "head_pet": y_pet[idx_train], "head_pa": y_pa[idx_train], "head_pc": y_pc[idx_train],
        "final_logic": y_raw[idx_train]
    }
    test_outputs = {
        "head_pet": y_pet[idx_test], "head_pa": y_pa[idx_test], "head_pc": y_pc[idx_test],
        "final_logic": y_raw[idx_test]
    }

    # 3. 构建物理模型
    print("🏗️ 构建物理感知模型...")
    # 关键：确保这里调用时，build_physics_model 已经定义好了
    model = build_physics_model(input_shape=(X.shape[1], 3), prior_weights=prior_matrix)

    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss={
            "head_pet": "binary_crossentropy",
            "head_pa": "binary_crossentropy",
            "head_pc": "binary_crossentropy",
            "final_logic": None
        },
        loss_weights={"head_pet": 1.0, "head_pa": 0.8, "head_pc": 0.8, "final_logic": 0.0},
        metrics={"head_pet": "accuracy", "head_pa": "accuracy", "head_pc": "accuracy"}
    )

    # 4. 训练
    print("🔥 开始训练...")
    history = model.fit(
        X_train, train_outputs,
        validation_data=(X_test, test_outputs),
        epochs=100,
        batch_size=256,
        callbacks=[
            callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    # 5. 评估与保存
    print("\n📊 最终评估...")
    preds = model.predict(X_test)
    final_pred_ids = preds[3].flatten()

    print(classification_report(y_raw[idx_test], final_pred_ids, target_names=["PET", "PA", "CC", "PC"]))

    h5_path = os.path.join(MODEL_SAVE_DIR, "physics_model.h5")
    model.save(h5_path)
    print(f"✅ H5 模型已保存: {h5_path}")

    onnx_path = os.path.join(MODEL_SAVE_DIR, "physics_model.onnx")
    spec = (tf.TensorSpec((None, X.shape[1], 3), tf.float32, name="physics_input"),)
    try:
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        with open(onnx_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        print(f"✅ ONNX 模型已导出: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX 导出失败: {e}")