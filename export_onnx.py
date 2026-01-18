import tensorflow as tf
import tf2onnx
import os
import json

# 配置路径
BASE_DIR = r"D:\DRL\DRL1"  # 你的工作目录
MODEL_PATH = os.path.join(BASE_DIR, 'final_model.h5')
CONFIG_FILE = "best_bands_config.json"
OUTPUT_ONNX_PATH = os.path.join(BASE_DIR, "pet_classifier_multiclass.onnx")

# 1. 读取配置获取波段数
with open("best_bands_config.json", 'r') as f:
    bands = json.load(f).get("all_unique_bands", [])
    num_bands = len(bands)

model = tf.keras.models.load_model('final_model.h5')

print(f"ℹ️ 模型输入波段数: {num_bands}")

# 2. 加载模型
model = tf.keras.models.load_model(MODEL_PATH)

# 3. 定义输入签名 (Signature)
# ⚠️ 关键修改：这里的 shape 必须是 num_bands
spec = (tf.TensorSpec((None, num_bands), tf.float32, name="input"),)


# 4. 转换
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=OUTPUT_ONNX_PATH
)

print(f"✅ ONNX 模型导出成功: {OUTPUT_ONNX_PATH}")