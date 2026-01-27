import os
import numpy as np
import spectral.io.envi as envi
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ================= 🔧 配置区域 (调试参数) =================

# 1. 输入文件路径 (修改为您的一张实际 .hdr 文件路径)
TEST_FILE_PATH = r"E:\SPEDATA\高谱相机数据集\训练集\PET\2025-11-5-15-20-6-894.hdr"

# 2. 纹理掩膜参数 (针对条纹背景)
# 阈值越小越严格，越容易把不均匀的区域当做背景
TEXTURE_THRESHOLD = 0.15

# 3. 二值掩膜参数 (针对黑色/暗色背景)
# 阈值越大越严格，越容易把暗的区域当做背景
# 【关键】低于此值的像素在图3中会被过滤掉
BINARY_THRESHOLD = 0.25


# ================= 🛠️ 核心函数库 =================

def generate_pca_texture_mask(img_data, diff_threshold=0.35):
    """
    方法A: 基于 PCA 主成分的纹理掩膜 (Texture Mask)
    原理: 利用 PCA 提取图像中"最显著的结构"，不依赖特定波段。
    返回: (mask, anisotropy_map, pc1_img)
    """
    try:
        H, W, B = img_data.shape

        # --- 1. PCA 提取主成分 (提取结构信息) ---
        flat_data = img_data.reshape(-1, B)
        # 降采样加速
        sample_indices = np.random.choice(flat_data.shape[0], min(10000, flat_data.shape[0]), replace=False)
        pca = PCA(n_components=1)
        pca.fit(flat_data[sample_indices])
        pc1 = pca.transform(flat_data).reshape(H, W)

        # 归一化到 0-1
        norm_pc1 = (pc1 - np.min(pc1)) / (np.max(pc1) - np.min(pc1) + 1e-6)

        # --- 2. 计算纹理各向异性 ---
        grad_x = cv2.Sobel(norm_pc1, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(norm_pc1, cv2.CV_64F, 0, 1, ksize=3)

        kernel = np.ones((5, 5), np.float32) / 25
        g_x = cv2.filter2D(np.abs(grad_x), -1, kernel)
        g_y = cv2.filter2D(np.abs(grad_y), -1, kernel)

        # 各向异性指数 (0=各向同性/材质, 1=强方向性/条纹)
        anisotropy = np.abs(g_x - g_y) / (g_x + g_y + 1e-6)

        # --- 3. 生成掩膜 ---
        # 只要纹理比较乱(anisotropy < 阈值) 就是材质
        is_material = (anisotropy < diff_threshold)

        # 形态学去噪
        mask = _apply_morphology(is_material)

        return mask, anisotropy, norm_pc1

    except Exception as e:
        print(f"⚠️ PCA 纹理掩膜计算出错: {e}")
        return np.zeros((img_data.shape[0], img_data.shape[1]), dtype=bool), None, None


def generate_binary_mask(img_data, brightness_threshold=0.15):
    """
    方法B: 基于亮度的二值掩膜 (Binary Mask)
    原理: 区分"亮物体"和"暗背景" (最简单，但对条纹背景可能无效)
    返回: (mask, intensity_map)
    """
    # 计算全波段平均亮度
    intensity = np.mean(img_data, axis=2)

    # 归一化到 0-1
    norm_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-6)

    # 生成掩膜: 够亮就是材质
    is_material = (norm_intensity > brightness_threshold)

    # 形态学去噪
    mask = _apply_morphology(is_material)

    return mask, norm_intensity


def _apply_morphology(mask_bool):
    """辅助函数: 形态学闭运算填补空洞"""
    mask_uint8 = mask_bool.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 开运算去掉噪点
    mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    # 闭运算填补内部空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def load_simple_envi(hdr_path):
    """简单的 ENVI 加载器"""
    if not os.path.exists(hdr_path): return None
    try:
        # 自动修复 header
        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        content = "".join(lines).lower()
        if "byte order" not in content:
            with open(hdr_path, 'a') as f: f.write("\nbyte order = 0")

        base = os.path.splitext(hdr_path)[0]
        spe_path = base + ".spe"
        if not os.path.exists(spe_path): spe_path = base + ".raw"

        img = envi.open(hdr_path, spe_path).load()
        if img.shape[1] < img.shape[2] and img.shape[1] in [208, 224]:
            img = np.transpose(img, (0, 2, 1))
        return np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None


# ================= 🚀 主程序 =================

if __name__ == "__main__":
    print(f"📂 正在加载: {TEST_FILE_PATH}")
    img_data = load_simple_envi(TEST_FILE_PATH)

    if img_data is not None:
        print(f"✅ 数据形状: {img_data.shape}")
        H, W, B = img_data.shape

        # 1. 计算纹理掩膜 (PCA版)
        print("🔄 计算 PCA 纹理掩膜...")
        tex_mask, tex_map, pc1 = generate_pca_texture_mask(img_data, diff_threshold=TEXTURE_THRESHOLD)

        # 2. 计算二值掩膜 (亮度版)
        print("🔄 计算二值亮度掩膜...")
        bin_mask, intensity_map = generate_binary_mask(img_data, brightness_threshold=BINARY_THRESHOLD)

        # 3. 计算交集掩膜 (最严格)
        combined_mask = tex_mask & bin_mask

        # --- [新增] 制作过滤后的 Intensity 图 (仅用于显示) ---
        # 只有大于阈值的区域保留原值，小于阈值的设为 0 (黑色)
        intensity_filtered = intensity_map.copy()
        intensity_filtered[intensity_filtered <= BINARY_THRESHOLD] = 0

        # --- 可视化对比 (2行3列) ---
        plt.figure(figsize=(18, 10))
        plt.suptitle(f"Mask Comparison: Texture(PCA) vs Binary(Intensity)", fontsize=16)

        # --- 第一行：特征图 ---

        # 1.1 PCA 主成分图
        plt.subplot(2, 3, 1)
        plt.title("1. PCA Structure (PC1)\n(Shows structural edges)")
        plt.imshow(pc1, cmap='gray')
        plt.axis('off')

        # 1.2 纹理热力图
        plt.subplot(2, 3, 2)
        plt.title("2. Texture Heatmap\n(Bright = High Anisotropy/Stripe)")
        plt.imshow(tex_map, cmap='jet')
        plt.colorbar(fraction=0.046)
        plt.axis('off')

        # 1.3 亮度热力图 (已添加阈值过滤)
        plt.subplot(2, 3, 3)
        plt.title(f"3. Intensity (Filtered > {BINARY_THRESHOLD})\n(Background Removed)")
        plt.imshow(intensity_filtered, cmap='inferno')  # inferno 配色对亮度更直观
        plt.colorbar(fraction=0.046)
        plt.axis('off')

        # --- 第二行：掩膜结果 ---

        # 2.1 纹理掩膜结果
        plt.subplot(2, 3, 4)
        plt.title(f"4. Texture Mask (PCA)\n(Thresh={TEXTURE_THRESHOLD})")
        plt.imshow(tex_mask, cmap='gray')
        plt.axis('off')

        # 2.2 二值掩膜结果
        plt.subplot(2, 3, 5)
        plt.title(f"5. Binary Mask (Intensity)\n(Thresh={BINARY_THRESHOLD})")
        plt.imshow(bin_mask, cmap='gray')
        plt.axis('off')

        # 2.3 最终交集掩膜
        plt.subplot(2, 3, 6)
        plt.title("6. Combined Mask\n(Intersection of 4 & 5)")
        plt.imshow(combined_mask, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print("\n💡 分析指南:")
        print("------------------------------------------------")
        print("图3 (Filtered Intensity): 现在只显示超过亮度阈值的区域，低于阈值的背景强制为黑色。")
        print("   -> 观察此图可以直观判断 BINARY_THRESHOLD 是否切掉了太多材质边缘。")
        print("------------------------------------------------")
        print("图6 (Combined Mask): 最终用于清洗数据的掩膜 (推荐)。")