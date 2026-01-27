import tensorflow as tf
import tf2onnx
import os

# 1. 加载训练好的 H5 模型
BASE_DIR = r"I:\Hyperspectral Camera Dataset\Processed_Data"
model_path = os.path.join(BASE_DIR, 'pet_transformer_final1.h5')
model = tf.keras.models.load_model(model_path)

# 2. 定义输入签名
# 这里的 30 对应你 DQN 选出的波段数量
# name="input" 必须指定，C++ 推理时会用到这个输入节点名称
spec = (tf.TensorSpec((None, 30), tf.float32, name="input"),)

# 3. 转换并保存
# opset 建议设为 13 或 15，以支持 MultiHeadAttention 算子
output_onnx_path = os.path.join(BASE_DIR, "pet_transformer_30bands.onnx")

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=output_onnx_path
)

print(f"✅ ONNX 模型已保存至: {output_onnx_path}")