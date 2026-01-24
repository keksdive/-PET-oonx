import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_spectral_curves(X, y, selected_bands, save_path="spectral_analysis.png"):
    """
    绘制论文级的光谱曲线与波段选择图 (类似 Fig. 10/11)

    参数:
    - X: 光谱数据 (N, Bands)
    - y: 标签 (0: Non-PET, 1: PET, 2: PA)
    - selected_bands: 算法选出的波段索引列表
    """
    print("🎨 正在生成光谱分析图...")

    # 定义类别名称和颜色
    # 假设 save_data.py 中: 0=Non-PET(背景), 1=PET(目标), 2=PA(困难负样本)
    class_info = {
        0: {"name": "Background/Other", "color": "#bdc3c7", "style": "--"},  # 灰色虚线
        1: {"name": "PET (Target)", "color": "#e74c3c", "style": "-"},  # 红色实线 (重点)
        2: {"name": "PA (Hard Neg)", "color": "#3498db", "style": "-."}  # 蓝色点划线
    }

    plt.figure(figsize=(12, 6), dpi=300)

    # 1. 绘制平均光谱曲线
    bands_x = np.arange(X.shape[1])

    for label_id, info in class_info.items():
        # 提取该类别的所有样本
        indices = np.where(y == label_id)[0]
        if len(indices) == 0:
            continue

        # 计算平均光谱
        mean_spectrum = np.mean(X[indices], axis=0)

        # 绘制曲线
        plt.plot(bands_x, mean_spectrum,
                 label=info["name"],
                 color=info["color"],
                 linestyle=info["style"],
                 linewidth=2 if label_id == 1 else 1.5)  # PET 线宽一点

        # 可选：绘制标准差阴影 (Standard Deviation Shadow)
        std_spectrum = np.std(X[indices], axis=0)
        plt.fill_between(bands_x,
                         mean_spectrum - 0.2 * std_spectrum,
                         mean_spectrum + 0.2 * std_spectrum,
                         color=info["color"], alpha=0.1)

    # 2. 绘制被选中的波段 (垂直条)
    # 使用灰色背景条表示选中的位置
    for band in selected_bands:
        plt.axvline(x=band, color='#2ecc71', linestyle='-', alpha=0.3, linewidth=1)
        # 或者使用 axvspan 画出有宽度的条
        # plt.axvspan(band-0.5, band+0.5, color='gray', alpha=0.3)

    # 3. 图表美化
    plt.title(f"Spectral Signature & Selected Bands (Count: {len(selected_bands)})", fontsize=14)
    plt.xlabel("Spectral Band Index", fontsize=12)
    plt.ylabel("Normalized Reflectance", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, X.shape[1])
    plt.ylim(0, 1.0)  # 归一化数据通常在0-1之间

    # 4. 标注 "Selected Bands" 字样 (模仿论文图例)
    # 在图的左上角画一个小矩形作为图例补充
    plt.text(5, 0.95, f"| Green Lines: Selected Features",
             color='#2ecc71', fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 图表已保存: {save_path}")
    # plt.show() # 如果在服务器上运行请注释掉