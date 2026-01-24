import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from q_network import QNetwork  # 保持引用你的 QNetwork 模块
import random
from collections import deque


class BandSelectionAgent:
    def __init__(self, num_bands=256):
        self.num_bands = num_bands
        self.gamma = 0.99  # 折扣因子
        self.batch_size = 64  # 稍微减小 Batch Size 以适应波段选择的稀疏性
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.001  # 最小探索率
        self.epsilon_decay = 0.995  # 衰减率

        # 经验回放池
        self.memory = deque(maxlen=50000)

        # --- 1. 初始化网络 ---
        self.model = QNetwork(num_bands)
        self.target_model = QNetwork(num_bands)

        # Build 模型 (输入维度: Batch x num_bands)
        dummy_input = tf.zeros((1, num_bands))
        self.model(dummy_input)
        self.target_model(dummy_input)

        # --- 2. 优化器配置 (关键优化) ---
        # 使用 Huber Loss 代替 MSE，防止梯度爆炸
        # 使用 clipnorm 防止梯度过大
        self.optimizer = optimizers.Nadam(learning_rate=1e-4, clipnorm=1.0)
        self.loss_fn = losses.Huber()

        # 初始化 Target 网络权重
        self.update_target_network()

    def update_target_network(self):
        """将主网络的权重复制到目标网络"""
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, selected_bands):
        """
        Epsilon-Greedy 策略选择波段
        """
        # 1. 探索：随机选择
        if np.random.rand() <= self.epsilon:
            available = [i for i in range(self.num_bands) if i not in selected_bands]
            return random.choice(available)

        # 2. 利用：选择 Q 值最大的
        # 将 state 转换为 (1, num_bands)
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        q_values = self.model(state_tensor).numpy()[0]

        # Masking: 将已选波段的 Q 值设为负无穷，防止重复选择
        for idx in selected_bands:
            q_values[idx] = -float('inf')

        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """
        训练主逻辑：升级为 Double DQN
        """
        if len(self.memory) < self.batch_size:
            return

        # 1. 随机采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # --- Double DQN 核心逻辑 ---

        # A. 使用 [当前网络] 预测下一状态的最佳动作 (argmax)
        next_q_online = self.model.predict(next_states, verbose=0)
        best_actions = np.argmax(next_q_online, axis=1)

        # B. 使用 [Target 网络] 评估该动作的 Q 值
        next_q_target = self.target_model.predict(next_states, verbose=0)

        # 提取 best_actions 对应的 Q 值
        rows = np.arange(self.batch_size)
        target_q_values = next_q_target[rows, best_actions]

        # C. 计算最终的目标 Q 值 (Bellman 方程)
        # target = r + gamma * Q_target(s', argmax Q_online(s', a))
        targets = rewards + self.gamma * target_q_values * (1 - dones)

        # --- 梯度下降更新 ---
        with tf.GradientTape() as tape:
            # 获取当前状态的 Q 值预测
            current_q_values = self.model(states, training=True)

            # 我们只关心 Agent 实际采取的那个 action 对应的 Q 值
            # 使用 one-hot 编码提取对应 action 的 Q 值
            one_hot_actions = tf.one_hot(actions, self.num_bands)
            predicted_q = tf.reduce_sum(current_q_values * one_hot_actions, axis=1)

            # 计算 Huber Loss
            loss = self.loss_fn(targets, predicted_q)

        # 反向传播
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)