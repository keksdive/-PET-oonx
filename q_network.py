import tensorflow as tf
from tensorflow.keras import layers, models


class QNetwork(tf.keras.Model):
    def __init__(self, num_bands=208):
        super(QNetwork, self).__init__()
        # 共享特征提取层
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.1)  # 增加轻量级随机扰动，提升鲁棒性

        # --- Dueling 结构核心 ---
        # 1. 状态价值流 (Value Stream): 判断当前已选波段的组合好不好
        self.v_dense = layers.Dense(128, activation='relu')
        self.v_out = layers.Dense(1)

        # 2. 动作优势流 (Advantage Stream): 判断下一步选哪个波段更好
        self.a_dense = layers.Dense(128, activation='relu')
        self.a_out = layers.Dense(num_bands)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dropout(x)

        # 计算 Value 和 Advantage
        v = self.v_dense(x)
        v = self.v_out(v)

        a = self.a_dense(x)
        a = self.a_out(a)

        # 结合：Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # 这种减去均值的操作能增加训练的稳定性
        q_values = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))

        return q_values