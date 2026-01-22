import os
import json
import subprocess
import numpy as np

# ================= 配置区域 =================
CONFIG_FILE = "best_bands_config.json"
# 假设 main.py 是跑 RL 选波段的脚本
RL_SCRIPT = "main.py"
# 假设 train_transformer.py 是训练 Transformer 的脚本
TRAIN_SCRIPT = "train_transformer.py"


def step_1_select_bands():
    print("\n🚀 [Step 1] 启动 DRL 智能体进行波段挑选...")
    # 运行你的强化学习主程序
    # 你需要修改 main.py，使其在结束后将最优波段列表保存到 JSON
    subprocess.run(["python", RL_SCRIPT], check=True)

    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError("❌ DRL 训练未生成配置文件，请检查 main.py 是否保存了结果！")

    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    bands = config.get("selected_bands", [])
    print(f"✅ 波段挑选完成！共选中 {len(bands)} 个波段: {bands}")
    return bands


def step_2_train_and_export():
    print("\n🚀 [Step 2] 启动 Transformer 分类器训练 & ONNX 导出...")
    # 调用训练脚本，训练脚本内部应该去读取 CONFIG_FILE
    subprocess.run(["python", TRAIN_SCRIPT], check=True)


if __name__ == "__main__":
    print("=" * 50)
    print("   全自动高光谱 AI 流水线 (Auto-HSI-Pipeline)")
    print("=" * 50)

    try:
        # 1. 挑选波段
        best_bands = step_1_select_bands()

        # 2. 训练模型 (包含自动导出 ONNX)
        step_2_train_and_export()

        print("\n🎉🎉🎉 全流程执行成功！模型已就绪。")

    except Exception as e:
        print(f"\n❌ 流程中断: {e}")