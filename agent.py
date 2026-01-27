import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import random
import os


# ================= 1. 子空间专家智能体 (The Specialist) =================
class SubSpaceAgent:
    def __init__(self, agent_id, scope_ranges, state_size, learning_rate=1e-4):
        self.id = agent_id
        self.scope_ranges = scope_ranges
        self.state_size = state_size

        # 构建局部动作空间映射
        self.local_to_global_map = []
        for start, end in scope_ranges:
            valid_end = min(end, state_size)
            if start < valid_end:
                self.local_to_global_map.extend(range(start, valid_end))

        self.action_size = len(self.local_to_global_map)

        if self.action_size == 0:
            print(f"⚠️ Warning: Agent {agent_id} has no valid bands in range!")

        # 神经网络
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # 优化器 - 将 learning_rate 保存为变量以便后续更新
        self.lr_schedule = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        self.optimizer = optimizers.Adam(learning_rate=self.lr_schedule, clipnorm=1.0)
        self.loss_fn = tf.keras.losses.Huber()

    def _build_model(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        v = layers.Dense(64, activation='relu')(x)
        v = layers.Dense(1)(v)
        a = layers.Dense(64, activation='relu')(x)
        a = layers.Dense(self.action_size)(a)
        q_vals = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
        return models.Model(inputs=inputs, outputs=q_vals)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_lr(self, new_lr):
        """🚀 动态更新学习率"""
        self.lr_schedule.assign(new_lr)

    def get_q_values(self, state):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        q_values = self.model(state_tensor, training=False)
        return q_values.numpy()[0]

    def train_step(self, states, actions_local, targets):
        with tf.GradientTape() as tape:
            q_values = self.model(states)
            indices = tf.stack([tf.range(tf.shape(actions_local)[0]), actions_local], axis=1)
            q_action = tf.gather_nd(q_values, indices)
            loss = self.loss_fn(targets, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


# ================= 2. 全局决策管理器 (The Fusion Brain) =================
class MultiAgentManager:
    def __init__(self, state_size=416, learning_rate=1e-4):
        self.state_size = state_size
        self.current_lr = learning_rate  # 记录当前LR

        # ⚠️ 根据实际特征维度调整
        self.agents = {
            "VNIR": SubSpaceAgent("VNIR", [(0, 70), (208, 278)], state_size, learning_rate),
            "SWIR1": SubSpaceAgent("SWIR1", [(70, 140), (278, 348)], state_size, learning_rate),
            "SWIR2": SubSpaceAgent("SWIR2", [(140, 208), (348, 416)], state_size, learning_rate)
        }

        self.memory = []
        self.memory_capacity = 5000
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.95

    # --- 新增: 更新所有子智能体的学习率 ---
    def update_learning_rate(self, new_lr):
        self.current_lr = new_lr
        for agent in self.agents.values():
            agent.update_lr(new_lr)

    # ------------------------------------

    def get_action(self, state, selected_bands):
        mask = np.zeros(self.state_size)
        for b in selected_bands:
            if b < self.state_size:
                mask[int(b)] = 1
        return self.get_global_action(state, mask)

    def train(self):
        return self.train_jointly()

    def update_target_network(self):
        self.update_targets()

    def get_global_action(self, state, selected_bands_mask):
        if np.random.rand() <= self.epsilon:
            available_indices = np.where(selected_bands_mask == 0)[0]
            if len(available_indices) == 0: return None
            return np.random.choice(available_indices)

        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        best_global_action = -1
        max_q_value = -float('inf')

        for name, agent in self.agents.items():
            if agent.action_size == 0: continue
            q_values = agent.get_q_values(state_tensor)
            for local_idx, global_idx in enumerate(agent.local_to_global_map):
                if selected_bands_mask[global_idx] == 1:
                    q_values[local_idx] = -float('inf')
            best_local_idx = np.argmax(q_values)
            best_local_q = q_values[best_local_idx]

            if best_local_q > max_q_value and best_local_q != -float('inf'):
                max_q_value = best_local_q
                best_global_action = agent.local_to_global_map[best_local_idx]

        if best_global_action == -1:
            available = np.where(selected_bands_mask == 0)[0]
            return np.random.choice(available) if len(available) > 0 else 0

        return best_global_action

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train_jointly(self):
        if len(self.memory) < self.batch_size: return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([m[0] for m in minibatch])
        actions_global = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        for name, agent in self.agents.items():
            agent_indices = []
            local_actions = []
            global_set = set(agent.local_to_global_map)
            for i, act in enumerate(actions_global):
                if act in global_set:
                    agent_indices.append(i)
                    local_actions.append(agent.local_to_global_map.index(act))
            if not agent_indices: continue

            s_batch = tf.convert_to_tensor(states[agent_indices], dtype=tf.float32)
            a_batch = tf.convert_to_tensor(local_actions, dtype=tf.int32)
            r_batch = tf.convert_to_tensor(rewards[agent_indices], dtype=tf.float32)
            ns_batch = tf.convert_to_tensor(next_states[agent_indices], dtype=tf.float32)
            d_batch = tf.convert_to_tensor(dones[agent_indices], dtype=tf.float32)

            next_q_online = agent.model(ns_batch)
            next_acts = tf.argmax(next_q_online, axis=1, output_type=tf.int32)
            next_q_target = agent.target_model(ns_batch)
            indices = tf.stack([tf.range(tf.shape(next_acts)[0]), next_acts], axis=1)
            target_vals = tf.gather_nd(next_q_target, indices)
            targets = r_batch + (1 - d_batch) * self.gamma * target_vals
            agent.train_step(s_batch, a_batch, targets)

    def update_targets(self):
        for agent in self.agents.values():
            agent.update_target_model()

    def save_model(self, filepath):
        base_dir = os.path.dirname(filepath)
        if base_dir and not os.path.exists(base_dir): os.makedirs(base_dir)
        for name, agent in self.agents.items():
            save_name = f"{filepath}_{name}.h5"
            agent.model.save_weights(save_name)

    def load_model(self, filepath):
        for name, agent in self.agents.items():
            load_name = f"{filepath}_{name}.h5"
            if os.path.exists(load_name):
                agent.model.load_weights(load_name)
                agent.update_target_model()
                print(f"Loaded weights for {name}")


BandSelectionAgent = MultiAgentManager