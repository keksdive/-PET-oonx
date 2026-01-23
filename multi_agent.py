import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import random
from collections import deque


class QNetwork(tf.keras.Model):
    """
    基础 Q 网络结构
    输入: 全局状态 (State Mask)
    输出: 局部动作空间的 Q 值 (Local Q-values)
    """

    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.d1 = layers.Dense(256, activation='relu')
        self.d2 = layers.Dense(256, activation='relu')
        self.d3 = layers.Dense(128, activation='relu')
        # 加入 Attention 机制 (改进点二的轻量化实现)
        self.attn = layers.Dense(128, activation='sigmoid')
        self.out = layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)

        # 简单的 Self-Attention 模拟
        a = self.attn(x)
        x = x * a

        x = self.d3(x)
        return self.out(x)


class SubspaceAgent:
    """
    子空间智能体：只负责特定波段范围的决策
    """

    def __init__(self, global_num_bands, action_range, name="Agent"):
        self.name = name
        self.global_num_bands = global_num_bands
        self.action_start, self.action_end = action_range
        self.num_local_actions = self.action_end - self.action_start

        self.gamma = 0.99
        self.batch_size = 64
        self.memory = deque(maxlen=20000)  # 每个智能体有自己的经验池

        # 网络初始化
        self.model = QNetwork(self.num_local_actions)
        self.target_model = QNetwork(self.num_local_actions)

        # Build
        dummy = tf.zeros((1, global_num_bands))
        self.model(dummy)
        self.target_model(dummy)

        self.optimizer = optimizers.Nadam(learning_rate=1e-4, clipnorm=1.0)
        self.loss_fn = losses.Huber()
        self.update_target_network()

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_local_q_values(self, state):
        """获取当前状态下，该智能体负责区域的所有 Q 值"""
        state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        return self.model(state_tensor).numpy()[0]

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)  # 这里是 Local Action Index

        # Double DQN Logic
        next_q_online = self.model.predict(next_states, verbose=0)
        best_local_actions = np.argmax(next_q_online, axis=1)

        next_q_target = self.target_model.predict(next_states, verbose=0)
        rows = np.arange(self.batch_size)
        target_q_values = next_q_target[rows, best_local_actions]

        targets = rewards + self.gamma * target_q_values * (1 - dones)

        with tf.GradientTape() as tape:
            current_q = self.model(states, training=True)
            one_hot = tf.one_hot(actions, self.num_local_actions)
            pred = tf.reduce_sum(current_q * one_hot, axis=1)
            loss = self.loss_fn(targets, pred)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss.numpy()


class MultiAgentManager:
    """
    多智能体管理器：负责分发任务、融合决策
    """

    def __init__(self, total_bands, ranges):
        """
        ranges: list of tuples, e.g., [(0, 70), (70, 140), (140, 208)]
        """
        self.agents = []
        self.ranges = ranges
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        for i, r in enumerate(ranges):
            agent = SubspaceAgent(total_bands, r, name=f"Agent_{i}_Range_{r[0]}-{r[1]}")
            self.agents.append(agent)
            print(f"🤖 初始化智能体: {agent.name} (Actions: {agent.num_local_actions})")

    def get_global_action(self, state, selected_bands):
        # 1. 探索
        if np.random.rand() <= self.epsilon:
            # 随机从所有未选波段中选一个
            available = [b for b in range(state.shape[0]) if b not in selected_bands]
            return random.choice(available)

        # 2. 利用 (协同决策)
        best_global_action = -1
        max_q_value = -float('inf')

        # 询问每个智能体
        for agent in self.agents:
            local_qs = agent.get_local_q_values(state)

            # Masking (在局部 Q 值中屏蔽已选波段)
            for global_idx in selected_bands:
                if agent.action_start <= global_idx < agent.action_end:
                    local_idx = global_idx - agent.action_start
                    local_qs[local_idx] = -float('inf')

            # 找出该智能体的最佳建议
            local_best_idx = np.argmax(local_qs)
            q_val = local_qs[local_best_idx]

            # 竞争：谁的 Q 值大听谁的
            if q_val > max_q_value:
                max_q_value = q_val
                best_global_action = agent.action_start + local_best_idx

        return best_global_action

    def remember(self, state, global_action, reward, next_state, done):
        # 将经验分发给负责该动作的智能体
        # 注意：只有执行了动作的智能体才需要学习，其他智能体这轮"轮空"
        for agent in self.agents:
            if agent.action_start <= global_action < agent.action_end:
                local_action = global_action - agent.action_start
                agent.memory.append((state, local_action, reward, next_state, done))
                break

    def train(self):
        losses = []
        for agent in self.agents:
            l = agent.train()
            losses.append(l)
        return losses

    def update_targets(self):
        for agent in self.agents:
            agent.update_target_network()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay