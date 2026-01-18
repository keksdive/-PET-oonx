import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import json
import os

# ================= 配置区域 =================
ONNX_PATH = "models/pet_classifier.onnx"
CONFIG_FILE = "best_bands_config.json"
# 验证集数据路径（建议与训练集分开，或使用训练时划分出的测试集）
VAL_DATA_DIR = r"D:\DRL\DRL1\data\val"

# 加载选中的波段
with open(CONFIG_FILE, 'r') as f:
    SELECTED_BANDS = json.load(f)["selected_bands"]


def run_validation():
    # 1. 加载 ONNX 模型
    print(f"正在加载模型: {ONNX_PATH}")
    session = ort.InferenceSession(ONNX_PATH)
    input_name = session.get_inputs()[0].name

    # 2. 准备验证数据 (参考 train_transformer.py 的加载逻辑)
    # 此处假设你已经准备好了 val_data.npy 和 val_mask.npy
    X_val = np.load(os.path.join(VAL_DATA_DIR, "val_data.npy"))[:, :, SELECTED_BANDS]
    y_val = np.load(os.path.join(VAL_DATA_DIR, "val_mask.npy"))

    # 展平数据
    H, W, C = X_val.shape
    X_flat = X_val.reshape(-1, C)
    y_flat = y_val.reshape(-1)

    # 过滤掉不需要验证的标签（例如只验证 PET(1) 和 非PET材质(2)+背景(0)）
    # 将标签统一为二分类：1=PET, 0=其他
    y_true = (y_flat == 1).astype(int)

    # 3. SNV 预处理
    print("应用 SNV 预处理...")
    mean = np.mean(X_flat, axis=1, keepdims=True)
    std = np.std(X_flat, axis=1, keepdims=True)
    X_flat_snv = (X_flat - mean) / (std + 1e-6)

    # 4. ONNX 推理
    print("开始 ONNX 推理...")
    # 注意：如果数据量极大，建议分 Batch 输入
    raw_preds = session.run(None, {input_name: X_flat_snv.astype(np.float32)})[0]
    y_pred = (raw_preds > 0.5).astype(int).flatten()

    # 5. 统计各项指标
    print("\n" + "=" * 30)
    print("📊 验证集性能报告")
    print("=" * 30)

    # 基础指标
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # Sensitivity 即 Recall
    f1 = f1_score(y_true, y_pred)

    # 计算 Specificity (特异度)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    print(f"Precision (精确率):   {precision:.4f}")
    print(f"Sensitivity (灵敏度): {sensitivity:.4f} (召回率)")
    print(f"Specificity (特异度): {specificity:.4f}")
    print(f"F1-Score:            {f1:.4f}")
    print("-" * 30)
    print("混淆矩阵:")
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")


if __name__ == "__main__":
    run_validation()