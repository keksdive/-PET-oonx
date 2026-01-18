import numpy as np  # 你原本只有 matplotlib，缺少 numpy

import matplotlib.pyplot as plt


def visualize_results(pure_pet_data, selected_bands):
    mean_spectrum = np.mean(pure_pet_data, axis=0)

    plt.figure(figsize=(12, 5))
    plt.plot(mean_spectrum, label='PET Mean Spectrum', color='k')
    for b in selected_bands:
        plt.axvline(x=b, color='r', alpha=0.3, linestyle='--')

    plt.title(f"Selected {len(selected_bands)} Bands Visualization")
    plt.xlabel("Band Index")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.show()