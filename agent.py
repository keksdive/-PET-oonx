import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from q_network import QNetwork
import random
from collections import deque


class BandSelectionAgent:
    def __init__(self, num_bands=256):
        self.num_bands = num_bands
        self.gamma = 0.99
        # ✅ 优化 1: 大幅增加 Batch Size 以发挥 3090 Ti 性能
        self.batch_size = 2048
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.memory = deque(maxlen=50000)

        self.model = QNetwork(num_bands)
        self.target_model = QNetwork(num_bands)

        dummy_input = tf.zeros((1, num_bands))
        self.model(dummy_input)
        self.target_model(dummy_input)

        # ✅ 优化 2: 提高学习率。更大的 Batch Size 通常可以配合略高的学习率
        self.optimizer = optimizers.Nadam(learning_rate=2e-4, clipnorm=1.0)
        self.loss_fn = losses.Huber()

        self.update_target_network()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, selected_bands):
        if np.random.rand() <= self.epsilon:
            available = [i for i in range(self.num_bands) if i not in selected_bands]
            return random.choice(available)

        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        # ✅ 优化 3: 使用 __call__ 代替 predict，并禁用训练模式
        q_values = self.model(state_tensor, training=False).numpy()[0]

        for idx in selected_bands:
            q_values[idx] = -float('inf')

        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # ✅ 优化 4: 使用 @tf.function 加速训练步 (Graph 模式)
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        # 1. 获取下一状态 Q 值
        next_q_online = self.model(next_states, training=False)
        best_actions = tf.argmax(next_q_online, axis=1)
        next_q_target = self.target_model(next_states, training=False)

        batch_idx = tf.range(tf.shape(actions)[0])
        indices = tf.stack([batch_idx, tf.cast(best_actions, tf.int32)], axis=1)
        target_q_values = tf.gather_nd(next_q_target, indices)

        # ✅ 修复：将所有参与贝尔曼方程计算的张量转为 float32
        target_q_values = tf.cast(target_q_values, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)

        # 计算贝尔曼目标 (r + gamma * Q_target)
        targets = rewards + self.gamma * target_q_values * (1 - dones)

        with tf.GradientTape() as tape:
            current_q_values = self.model(states, training=True)
            one_hot_actions = tf.one_hot(actions, self.num_bands)

            # 兼容混合精度：将 one_hot 转为模型权重相同的类型 (float16)
            one_hot_actions = tf.cast(one_hot_actions, current_q_values.dtype)
            predicted_q = tf.reduce_sum(current_q_values * one_hot_actions, axis=1)

            # ✅ 计算 Loss 时，将 targets 转为与模型输出一致的 float16
            loss = self.loss_fn(tf.cast(targets, predicted_q.dtype), predicted_q)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
        actions = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
        rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)
        dones = tf.convert_to_tensor(np.array(dones), dtype=tf.float32)

        return self.train_step(states, actions, rewards, next_states, dones)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)