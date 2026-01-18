import numpy as np  # 你原本只有 matplotlib，缺少 numpy

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_spectral_selection(X_data, y_data, selected_bands, save_path="selected_bands_analysis.png"):
    """
    可视化三种材质的平均光谱，并标记出 DRL 选择的波段位置。
    这能直观展示选中的波段是否位于材质差异最大的区域。
    """
    print("🎨 正在生成光谱分析图...")

    # 1. 分离各材质的数据
    # 假设标签: 0=Background, 1=PET, 2=CC, 3=PA
    # 注意：X_data 是 (N_samples, N_bands)

    pet_spectra = X_data[y_data == 1]
    cc_spectra = X_data[y_data == 2]
    pa_spectra = X_data[y_data == 3]

    # 2. 计算平均光谱 (Mean Spectrum)
    # 如果某类样本不存在，给一个全0数组防止报错
    mean_pet = np.mean(pet_spectra, axis=0) if len(pet_spectra) > 0 else np.zeros(X_data.shape[1])
    mean_cc = np.mean(cc_spectra, axis=0) if len(cc_spectra) > 0 else np.zeros(X_data.shape[1])
    mean_pa = np.mean(pa_spectra, axis=0) if len(pa_spectra) > 0 else np.zeros(X_data.shape[1])

    # 3. 开始绘图
    plt.figure(figsize=(15, 6))

    # 绘制材质曲线
    x_axis = np.arange(len(mean_pet))

    # 仅当数据存在时才绘制
    if len(pet_spectra) > 0:
        plt.plot(x_axis, mean_pet, color='red', label='PET (Target)', linewidth=2)
    if len(cc_spectra) > 0:
        plt.plot(x_axis, mean_cc, color='green', label='CC (Impurity)', linewidth=2, linestyle='--')
    if len(pa_spectra) > 0:
        plt.plot(x_axis, mean_pa, color='blue', label='PA (Impurity)', linewidth=2, linestyle='-.')

    # 4. 标记被选中的波段
    # 在底部画竖线，或者贯穿整图的背景条
    for band_idx in selected_bands:
        plt.axvline(x=band_idx, color='purple', alpha=0.2, linewidth=1)

    # 为了图例好看，只画一条模拟的“Selected Band”线
    plt.axvline(x=selected_bands[0], color='purple', alpha=0.5, linewidth=1, label='DRL Selected Bands')

    plt.title(f"Spectral Signature Analysis (Selected {len(selected_bands)} Bands)", fontsize=14)
    plt.xlabel("Band Index (Wavelength)", fontsize=12)
    plt.ylabel("Reflectance / Intensity", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 5. 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 光谱分析图已保存至: {os.path.abspath(save_path)}")
    plt.close()


def visualize_band_images(file_path, selected_bands, output_dir="band_visuals"):
    """
    (可选) 读取一张实际的 .npy 图片，展示选定波段的热力图
    这能让你看到在这些波段下，物体长什么样。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        data = np.load(file_path)  # (H, W, Bands)

        # 展示前 3 个被选中的波段
        display_bands = selected_bands[:3]

        plt.figure(figsize=(15, 5))
        for i, band_idx in enumerate(display_bands):
            plt.subplot(1, 3, i + 1)
            plt.imshow(data[:, :, band_idx], cmap='gray')
            plt.title(f"Selected Band: {band_idx}")
            plt.axis('off')

        save_path = os.path.join(output_dir, "sample_band_view.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ 样本波段可视化已保存至: {save_path}")

    except Exception as e:
        print(f"⚠️ 无法生成波段图像预览: {e}")