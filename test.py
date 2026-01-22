import tensorflow as tf
import os

# 强制显示详细日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

print("TensorFlow 版本:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ 成功检测到显卡:", gpus)
else:
    print("❌ 依然无法检测到显卡，请检查 CUDA 环境变量。")