import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_and_verify_pet_bands(X_data, y_data, selected_bands, save_path="band_selection_result.png"):
    """
    可视化波段选择结果，并验证 1600-1700nm 吸收峰覆盖情况
    """
    # 1. 提取 PET 样本的平均光谱 (Label 为 1)
    pet_indices = np.where(y_data == 1)[0]
    if len(pet_indices) == 0:
        print("⚠️ 未找到 PET 样本，无法生成波谱图。")
        return

    pet_mean_spectrum = np.mean(X_data[pet_indices], axis=0)
    num_bands = len(pet_mean_spectrum)

    # 2. 估算波长映射 (假设相机范围 935-1722nm，对应 208 个波段)
    # 根据文献 ，FX17 相机通常为 935.9-1722.5 nm
    start_wl, end_wl = 935.9, 1722.5
    wavelengths = np.linspace(start_wl, end_wl, num_bands)

    # 计算 1600-1700nm 对应的波段索引范围
    idx_1600 = np.argmin(np.abs(wavelengths - 1600))
    idx_1700 = np.argmin(np.abs(wavelengths - 1700))

    # 3. 验证覆盖情况
    covered_bands = [b for b in selected_bands if idx_1600 <= b <= idx_1700]
    print(f"\n🔍 [验证] 1600-1700nm (索引 {idx_1600}-{idx_1700}) 区域内选中了 {len(covered_bands)} 个波段。")
    if len(covered_bands) > 0:
        print(f"✅ 包含特征吸收峰波段: {covered_bands}")
    else:
        print("❌ 警告：选中的波段未覆盖 1600-1700nm 核心特征区，请检查数据或增加训练轮数。")

    # 4. 绘图
    plt.figure(figsize=(12, 6))

    # 绘制 PET 平均反射率曲线
    plt.plot(wavelengths, pet_mean_spectrum, label='PET Mean Spectrum', color='black', linewidth=2)

    # 高亮 1600-1700nm 区域 (吸收峰区域)
    plt.axvspan(1600, 1700, color='yellow', alpha=0.2, label='PET Peak Area (1600-1700nm)')

    # 标记选中的波段
    first_mark = True
    for b in selected_bands:
        wl = wavelengths[b]
        label = "Selected Bands" if first_mark else ""
        plt.axvline(x=wl, color='red', linestyle='--', alpha=0.4, label=label)
        first_mark = False

    plt.title(f"D3QN Band Selection Result (Total: {len(selected_bands)} bands)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Reflectance")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300)
    print(f"📊 结果图已保存至: {os.path.abspath(save_path)}")
    plt.show()