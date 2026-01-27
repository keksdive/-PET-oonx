import numpy as np
import os
import json
import tensorflow as tf
from agent import BandSelectionAgent
from reward_utils import calculate_hybrid_reward
from visualization import visualize_spectral_curves
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ================= 🔧 1. 全局配置 =================
DATA_DIR = r"D:\Train_Data\NP_new_MultiClass_SNV"

# 结果保存路径
JSON_SAVE_DIR = r"D:\Processed_Result\json-procession-result"
if not os.path.exists(JSON_SAVE_DIR): os.makedirs(JSON_SAVE_DIR)

CONFIG_FILENAME = os.path.join(JSON_SAVE_DIR, "material_specific_features.json")
FIGURE_DIR = "results_figures"
if not os.path.exists(FIGURE_DIR): os.makedirs(FIGURE_DIR)

# DRL 参数
NUM_BANDS_TO_SELECT = 30
TOTAL_EPISODES = 1200  # 适当调整轮数
SAMPLES_PER_CLASS = 2000

# 优化器策略
LEARNING_RATE_START = 1e-4
LEARNING_RATE_MIN = 1e-6
LR_DECAY_RATE = 0.99

# 💡 物理特征知识库
EXPERT_KNOWLEDGE = {
    "PET": {
        "label_id": 1,
        "description": "Polyethylene terephthalate",
        "features": [
            {"nm": 1660, "idx": 126, "topology": "Valley", "deriv_behavior": "ZeroCrossing_NegPos", "width": 3,
             "weight": 1.0},
            {"nm": 1129, "idx": 73, "topology": "Valley", "deriv_behavior": "LocalMin", "width": 3, "weight": 0.8},
            {"nm": 1170, "idx": 77, "topology": "Valley", "deriv_behavior": "LocalMin", "width": 3, "weight": 0.8}
        ],
        "exclusion_rules": []
    },
    "PA": {
        "label_id": 2,
        "description": "Polyamide (Nylon)",
        "features": [
            {"nm": 1520, "idx": 112, "topology": "Broad_Valley", "deriv_behavior": "Gentle_Slope", "width": 8,
             "weight": 1.0}
        ],
        "exclusion_rules": []
    },
    "PC": {
        "label_id": 4,
        "description": "Polycarbonate",
        "features": [
            {"nm": 1685, "idx": 129, "topology": "Valley_Shifted", "deriv_behavior": "Sharp_Peak", "width": 4,
             "weight": 1.0},
            {"nm": 1195, "idx": 80, "topology": "Valley", "deriv_behavior": "LocalMin", "width": 4, "weight": 0.8}
        ],
        "exclusion_rules": []
    },
    "CC": {
        "label_id": 3,
        "description": "Calcium Carbonate",
        "features": [],
        "topology_global": "Flat_High_Reflectance",
        "exclusion_rules": ["Exclude if PET_1660_Valley detected", "Exclude if PA_1520_BroadValley detected"]
    }
}

EXPERT_REWARD_WEIGHT = 0.8


# ================= 🛠️ 工具类：NumPy JSON 编码器 =================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# ================= 2. 辅助与验证函数 =================

def load_data():
    x_path = os.path.join(DATA_DIR, "X.npy")
    y_path = os.path.join(DATA_DIR, "y.npy")
    if not os.path.exists(x_path): raise Exception(f"Data not found in {DATA_DIR}")
    X = np.load(x_path, mmap_mode='r')
    y = np.load(y_path)
    return X, y


def prepare_binary_data(X_full, y_full, target_label_id, n_samples=2000):
    pos_indices = np.where(y_full == target_label_id)[0]
    if len(pos_indices) == 0: return None, None
    neg_indices = np.where(y_full != target_label_id)[0]
    n_pos = min(len(pos_indices), n_samples)
    n_neg = min(len(neg_indices), n_samples)
    pos_sel = np.random.choice(pos_indices, n_pos, replace=False)
    neg_sel = np.random.choice(neg_indices, n_neg, replace=False)
    X_bin = np.concatenate([X_full[pos_sel], X_full[neg_sel]])
    y_bin = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    perm = np.random.permutation(len(y_bin))
    return X_bin[perm], y_bin[perm]


def create_advanced_gravity_field(total_dim, features_info):
    field = np.zeros(total_dim, dtype=np.float32)
    is_stacked = (total_dim > 300)
    half_dim = total_dim // 2 if is_stacked else total_dim
    for feat in features_info:
        base_idx = feat['idx']
        width = feat['width']
        weight = feat.get('weight', 1.0)
        if base_idx < half_dim:
            s = max(0, base_idx - width)
            e = min(half_dim, base_idx + width + 1)
            field[s:e] = 1.0 * weight
        if is_stacked:
            d_idx = base_idx + half_dim
            if d_idx < total_dim:
                s = max(half_dim, d_idx - width)
                e = min(total_dim, d_idx + width + 1)
                field[s:e] = 1.0 * weight
    return field


def analyze_selection_logic(selected_bands, mat_info, total_dim=416):
    is_stacked = (total_dim > 300)
    half_dim = total_dim // 2 if is_stacked else total_dim
    report_items = []
    intensity_bands = set([b for b in selected_bands if b < half_dim])
    deriv_bands = set([b - half_dim for b in selected_bands if b >= half_dim])
    aligned_indices = intensity_bands.intersection(deriv_bands)

    for b in selected_bands:
        b_type = "Intensity" if b < half_dim else "Derivative"
        real_idx = b if b < half_dim else b - half_dim
        nm_approx = 400 + real_idx * 10
        match_feat = None
        for feat in mat_info.get('features', []):
            if abs(real_idx - feat['idx']) <= feat['width']:
                match_feat = feat
                break
        item = {
            "index": int(b), "type": b_type, "nm_approx": nm_approx,
            "physical_match": "None", "topology_expect": "Unknown", "derivative_expect": "Unknown",
            "alignment_status": "Single"
        }
        if match_feat:
            item["physical_match"] = f"Hit {match_feat['nm']}nm Feature"
            item["topology_expect"] = match_feat['topology']
            item["derivative_expect"] = match_feat['deriv_behavior']
            if real_idx in aligned_indices:
                item["alignment_status"] = "Aligned (Int+Deriv Selected)"
        report_items.append(item)
    return report_items


# --- [新增] 验证模块 ---
def validate_bands_performance(X_train, y_train, X_test, y_test, selected_bands, mat_name):
    """
    使用 SVM 代理分类器验证所选波段的有效性
    """
    print(f"   ⚖️ 正在验证波段有效性 (使用 SVM 代理评估)...")

    # 1. 仅保留选中的波段
    X_tr_sel = X_train[:, selected_bands]
    X_te_sel = X_test[:, selected_bands]

    # 2. 训练轻量级分类器 (SVM)
    clf = SVC(kernel='rbf', C=1.0, cache_size=500)
    clf.fit(X_tr_sel, y_train)

    # 3. 预测与评估
    y_pred = clf.predict(X_te_sel)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # 4. 绘制混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Other', mat_name], yticklabels=['Other', mat_name])
    plt.title(f'Validation Matrix: {mat_name}\n(Acc: {acc:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_path = os.path.join(FIGURE_DIR, f"Val_Matrix_{mat_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   📈 验证完成! 测试集准确率: {acc:.4f} | 混淆矩阵已保存: {save_path}")

    return acc, report


def plot_reward_curve(rewards, mat_name):
    """绘制 DRL 训练奖励曲线"""
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.6, label='Raw Reward')
    # 绘制平滑曲线
    window = max(5, len(rewards) // 20)
    if len(rewards) > window:
        smooth = np.convolve(rewards, np.ones(window) / window, mode='valid')
        plt.plot(smooth, color='red', linewidth=2, label='Smoothed')
    plt.title(f'DRL Training Convergence: {mat_name}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    save_path = os.path.join(FIGURE_DIR, f"Train_Curve_{mat_name}.png")
    plt.savefig(save_path)
    plt.close()


# ================= 3. 主流程 =================

def main():
    X_full, y_full = load_data()
    num_total_bands = X_full.shape[1]
    final_json = {
        "pipeline_info": {
            "description": "Physics-Augmented DRL Selection with Validation",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_dim": num_total_bands
        },
        "materials": {}
    }

    print(f"\n🚀 启动物理增强型特征筛选与验证流程...")

    for mat_name, info in EXPERT_KNOWLEDGE.items():
        label_id = info['label_id']
        print(f"\n{'=' * 60}")
        print(f"🔬 材质分析: {mat_name} (Label {label_id})")

        # 1. 准备二分类数据
        X_bin, y_bin = prepare_binary_data(X_full, y_full, label_id, SAMPLES_PER_CLASS)
        if X_bin is None:
            print(f"⚠️ 跳过 (数据集中无 Label {label_id})")
            continue

        # 2. [关键修改] 划分训练集和测试集 (80% 训练, 20% 验证)
        # DRL 只在 Train 上跑，结果在 Test 上验，保证泛化能力
        X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)
        print(f"   📦 数据划分: 训练集 {X_train.shape[0]} | 验证集 {X_test.shape[0]}")

        gravity = create_advanced_gravity_field(num_total_bands, info.get('features', []))

        # 初始化 Agent
        try:
            agent = BandSelectionAgent(num_total_bands, learning_rate=LEARNING_RATE_START)
        except:
            agent = BandSelectionAgent(num_total_bands)

        best_r = -float('inf')
        best_bands = []
        current_lr = LEARNING_RATE_START
        reward_history = []

        # 3. DRL 训练循环 (只使用 X_train)
        for e in range(TOTAL_EPISODES):
            state = np.zeros(num_total_bands)
            sel = []
            ep_r = 0

            for step in range(NUM_BANDS_TO_SELECT):
                action = agent.get_action(state, sel)

                # 奖励计算 (使用 X_train)
                r_dat = calculate_hybrid_reward(sel, action, X_train, y_train, alpha=2.0, beta=1.0)
                r_exp = EXPERT_REWARD_WEIGHT if gravity[action] > 0 else 0
                r_align = 0
                is_stacked = (num_total_bands > 300)
                if is_stacked:
                    half = num_total_bands // 2
                    pair_idx = action - half if action >= half else action + half
                    if pair_idx < num_total_bands and state[pair_idx] == 1:
                        r_align = 0.2

                r = r_dat + r_exp + r_align

                ns = state.copy()
                ns[action] = 1
                done = (step == NUM_BANDS_TO_SELECT - 1)
                agent.remember(state, action, r, ns, done)
                agent.train()

                state = ns
                sel.append(action)
                ep_r += r

            reward_history.append(ep_r)

            # 参数衰减
            if hasattr(agent, 'epsilon') and hasattr(agent, 'epsilon_decay'):
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

            new_lr = max(LEARNING_RATE_MIN, current_lr * LR_DECAY_RATE)
            if new_lr != current_lr:
                current_lr = new_lr
                if hasattr(agent, 'update_learning_rate'):
                    agent.update_learning_rate(current_lr)
                elif hasattr(agent, 'optimizer') and hasattr(agent.optimizer, 'learning_rate'):
                    agent.optimizer.learning_rate.assign(current_lr)

            if ep_r > best_r:
                best_r = ep_r
                best_bands = sorted(sel)

            if (e + 1) % 10 == 0:  # 减少打印频率
                print(f"   ...Ep {e + 1}/{TOTAL_EPISODES} | Best R: {best_r:.4f} | Eps: {agent.epsilon:.4f}")

        print(f"   ✅ DRL 筛选完成. 选出 {len(best_bands)} 个特征")

        # 4. [新增] 绘制训练曲线
        plot_reward_curve(reward_history, mat_name)

        # 5. [新增] 在测试集上验证泛化能力
        val_acc, val_report = validate_bands_performance(X_train, y_train, X_test, y_test, best_bands, mat_name)

        # 6. 生成报告
        analysis = analyze_selection_logic(best_bands, info, num_total_bands)
        final_json["materials"][mat_name] = {
            "selected_bands": [int(b) for b in best_bands],
            "validation_metrics": {
                "accuracy": val_acc,
                "report": val_report
            },
            "physics_metadata": {
                "features_defined": len(info.get('features', [])),
                "exclusion_rules": info.get('exclusion_rules', [])
            },
            "band_analysis": analysis
        }

        try:
            visualize_spectral_curves(X_bin, y_bin, selected_bands=best_bands,
                                      save_path=os.path.join(FIGURE_DIR, f"PhysSpec_{mat_name}.png"))
        except:
            pass

    # 保存结果
    with open(CONFIG_FILENAME, "w", encoding='utf-8') as f:
        json.dump(final_json, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\n💾 最终验证报告已保存: {CONFIG_FILENAME}")


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)
    main()