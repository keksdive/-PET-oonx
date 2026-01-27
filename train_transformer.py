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
DATA_DIR = r"G:\NP_new_MultiClass_SNV"
JSON_PATH = r"G:\多酚类\json-procession-result\material_specific_features.json"
MODEL_SAVE_DIR = r"G:\多酚类\final_cascade_model"
RESULT_DIR = r"G:\多酚类\json-procession-result"

if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)

# 显存配置
gpus = tf.config.list_physical_devices('GPU')
if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)


# ================= 2. 物理特征工程 (Physics-Aware Features) =================

def compute_physics_features(X):
    """
    [核心物理逻辑]
    将原始光谱转化为物理特征向量。
    1. 强度 (Intensity): 基础反射率
    2. 一阶导 (Slope): 区分上升/下降沿
    3. 二阶导 (Curvature): *关键* 区分 PET(尖峰) vs PA(宽峰) vs CC(平坦)
    """
    # 1. 强度 (原始)
    f0 = X

    # 2. 一阶导数 (斜率)
    f1 = np.gradient(f0, axis=1)

    # 3. 二阶导数 (曲率) - 让模型学会识别 "V型谷" 的锐度
    f2 = np.gradient(f1, axis=1)

    # 堆叠: (Batch, Bands, 3)
    # 这样模型在每个波段上都能同时看到：有多亮？在变亮还是变暗？是尖峰还是平底？
    X_stacked = np.stack([f0, f1, f2], axis=-1)
    return X_stacked


# ================= 3. 数据加载与标签编码 =================

def load_data():
    print("📥 加载数据...")
    X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.float32)

    # 假设 y 的 ID: 1=PET, 2=PA, 3=CC, 4=PC (如果您的数据里有PC)
    # 如果没有 PC，稍后逻辑层会处理

    # 读取 JSON 波段配置 (如果有)
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            config = json.load(f)
        selected_bands = set()
        for mat in config['materials'].values():
            selected_bands.update(mat['selected_bands'])
        selected_bands = sorted(list(selected_bands))
        X = X[:, selected_bands]
        print(f"🔪 物理波段切片: {len(selected_bands)} bands")

    # 计算物理特征 (Intensity + Slope + Curvature)
    print("🧠 计算物理特征 (二阶导数)...")
    X_physics = compute_physics_features(X)

    # 过滤背景 (ID 0)
    valid = y != 0
    X_physics = X_physics[valid]
    y = y[valid]

    # 多头标签编码
    y_pet = (y == 1).astype(np.float32)
    y_pa = (y == 2).astype(np.float32)
    y_pc = (y == 4).astype(np.float32)  # 假设 PC 是 ID 4，如果没有则全 0
    # CC (ID 3) 是通过排除法得到的

    return X_physics, y, y_pet, y_pa, y_pc


# ================= 4. 模型组件: 物理注意力与逻辑层 =================

def parse_physics_priors(json_path, sorted_selected_bands):
    """
    [核心升级] 深度解析 JSON，提取物理先验权重
    返回: prior_weights (Num_Bands, 3) -> [Intensity, Slope, Curvature]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 建立映射: 原始波段ID -> 切片后的索引
    band_map = {original_idx: i for i, original_idx in enumerate(sorted_selected_bands)}
    num_bands = len(sorted_selected_bands)

    # 初始化权重: 默认为 1.0 (平权)
    # 通道定义: 0=Intensity, 1=Slope(1st Deriv), 2=Curvature(2nd Deriv)
    priors = np.ones((num_bands, 3), dtype=np.float32)

    print("🧠 [Physics] 正在注入物理先验知识...")

    for mat_name, mat_data in data['materials'].items():
        if 'band_analysis' not in mat_data: continue

        for item in mat_data['band_analysis']:
            original_idx = item['index']
            if original_idx not in band_map: continue

            idx = band_map[original_idx]

            # --- 规则 1: 物理匹配 (Physical Match) ---
            # 如果明确命中了物理特征 (如 C-H 键)，大幅提升该波段所有通道的权重
            if "Hit" in item.get('physical_match', ''):
                priors[idx] *= 3.0  # 重点关注！

            # --- 规则 2: 特征类型 (Type) ---
            # 如果 JSON 说这个波段看的是 "Derivative" (导数)，则提升 Slope 和 Curvature 的权重
            if item.get('type') == 'Derivative':
                priors[idx, 1] *= 2.0  # 关注斜率
                priors[idx, 2] *= 2.0  # 关注曲率

            # --- 规则 3: 拓扑预期 (Topology) ---
            # 如果是波峰/波谷，说明曲率(二阶导)是关键特征
            topo = item.get('topology_expect', 'Unknown')
            if topo in ['Valley', 'Peak', 'LocalMin', 'LocalMax']:
                priors[idx, 2] *= 4.0  # 极度关注曲率！这是区分材质形状的关键

    # 归一化 (保持数值稳定性)
    priors = priors / np.mean(priors)
    return priors


class CascadeLogicLayer(layers.Layer):
    """
    [级联逻辑] PET -> PA -> PC -> Else CC
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs: [pet_prob, pa_prob, pc_prob]
        pet_p, pa_p, pc_p = inputs

        is_pet = tf.greater(pet_p, 0.5)
        is_pa = tf.greater(pa_p, 0.5)
        is_pc = tf.greater(pc_p, 0.5)

        # 逻辑树:
        # If PET -> 1
        # Else If PA -> 2
        # Else If PC -> 4 (注意您的ID定义)
        # Else -> 3 (CC)

        val_pet = tf.cast(1.0, tf.float32)
        val_pa = tf.cast(2.0, tf.float32)
        val_cc = tf.cast(3.0, tf.float32)
        val_pc = tf.cast(4.0, tf.float32)  # 假设 PC ID 为 4

        out = tf.where(is_pet, val_pet,
                       tf.where(is_pa, val_pa,
                                tf.where(is_pc, val_pc, val_cc)))
        return out


# ================= 5. 模型构建 =================

def build_physics_model(input_shape):
    # input_shape: (Bands, 3) -> 3个物理通道
    inputs = layers.Input(shape=input_shape, name="physics_input")

    # 1. 物理注意力层 (关注关键波段的曲率)
    x = PhysicsGuidedAttention(name="physics_attention")(inputs)

    # 2. 特征提取 (CNN 处理局部波形)
    x = layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)

    # 3. 独立决策头 (Specialists)
    # Head PET: 专看尖峰
    h_pet = layers.Dense(16, activation='relu')(x)
    out_pet = layers.Dense(1, activation='sigmoid', name="head_pet")(h_pet)

    # Head PA: 专看宽谷
    h_pa = layers.Dense(16, activation='relu')(x)
    out_pa = layers.Dense(1, activation='sigmoid', name="head_pa")(h_pa)

    # Head PC: 专看苯环特征
    h_pc = layers.Dense(16, activation='relu')(x)
    out_pc = layers.Dense(1, activation='sigmoid', name="head_pc")(h_pc)

    # 4. 逻辑层
    final_id = CascadeLogicLayer(name="final_logic")([out_pet, out_pa, out_pc])

    return models.Model(inputs=inputs, outputs=[out_pet, out_pa, out_pc, final_id])


# ================= 6. 主程序 =================

if __name__ == "__main__":
    # 1. 准备数据
    X, y_raw, y_pet, y_pa, y_pc = load_data()

    # 划分
    indices = np.arange(len(X))
    X_train, X_test, idx_train, idx_test = train_test_split(X, indices, test_size=0.2, stratify=y_raw, random_state=42)

    train_out = {"head_pet": y_pet[idx_train], "head_pa": y_pa[idx_train], "head_pc": y_pc[idx_train],
                 "final_logic": y_raw[idx_train]}
    test_out = {"head_pet": y_pet[idx_test], "head_pa": y_pa[idx_test], "head_pc": y_pc[idx_test],
                "final_logic": y_raw[idx_test]}

    # 2. 构建
    # 输入维度变为 (Bands, 3)
    model = build_physics_model((X.shape[1], 3))

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

    # 3. 训练
    print("🔥 开始训练物理感知模型...")
    model.fit(X_train, train_out, validation_data=(X_test, test_out), epochs=50, batch_size=256)

    # 4. 导出 ONNX (注意 Input Spec 变了)
    spec = (tf.TensorSpec((None, X.shape[1], 3), tf.float32, name="physics_input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec,
                               output_path=os.path.join(MODEL_SAVE_DIR, "physics_model.onnx"))
    print("✅ 完成")