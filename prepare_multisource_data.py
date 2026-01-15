import numpy as np
import os
import glob
from data_preprocessing import load_and_preprocess_data

# ================= 配置区域 =================
# 1. DQN 选出的 30 个波段索引
SELECTED_BANDS = [19, 39, 62, 69, 70, 72, 74, 76, 78, 83, 90, 93, 95, 103, 105, 106, 112, 115, 123, 128, 133, 140, 143, 150, 160, 172, 174, 180, 187, 197]

# 2. 路径配置
# [输入] 非 PET 材质的 .SPE/.HDR 文件夹
NON_PET_SPE_DIR = r"I:\Hyperspectral Camera Dataset\Train_Data\no_PET\no_PET(CC醋酸纤维素)"

# [输入] 34 个 .npy 格式数据的文件夹 (已禁用)
# NPY_DIR = r"I:\Hyperspectral Camera Dataset\Train_Data\no_PET_Processed_RL"

# [输入] 验证集文件夹
VAL_DIR = r"I:\Hyperspectral Camera Dataset\测试集\PET"

# [校准文件]
WHITE_REF = r"I:\Hyperspectral Camera Dataset\B_W\bai1.wcor"
DARK_REF = r"I:\Hyperspectral Camera Dataset\B_W\hei1.dcor"

# [输出] 处理结果保存路径
SAVE_DIR = r"I:\Hyperspectral Camera Dataset\Processed_Data"


# ===========================================

def fix_all_headers_in_folder(folder_path):
    """自动补全缺失的 byte order"""
    if not os.path.exists(folder_path):
        print(f"⚠️ 路径不存在，跳过修复: {folder_path}")
        return

    hdr_files = glob.glob(os.path.join(folder_path, "*.hdr"))
    print(f"🔧 正在检查 {folder_path} 下的 {len(hdr_files)} 个头文件...")

    count = 0
    for hdr_path in hdr_files:
        try:
            with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            if not any('byte order' in line.lower() for line in lines):
                with open(hdr_path, 'a') as f:
                    f.write('\nbyte order = 0')
                count += 1
        except Exception as e:
            print(f"  ❌ 修复 {os.path.basename(hdr_path)} 失败: {e}")

    if count > 0:
        print(f"  ✅ 已修复 {count} 个缺失 byte order 的头文件。")


def process_spe_folder(folder_path, label_name, threshold=0.01):
    """读取 SPE 文件夹，先修复头文件，再切片 30 波段"""
    print(f"\n📂 正在处理 {label_name} (.SPE)...")
    if not os.path.exists(folder_path):
        print(f"⚠️ 文件夹不存在: {folder_path}")
        return None

    fix_all_headers_in_folder(folder_path)

    try:
        # threshold 设低一点
        raw_data = load_and_preprocess_data(folder_path, WHITE_REF, DARK_REF, threshold=threshold)

        # 切片: (N, 208) -> (N, 30)
        reduced_data = raw_data[:, SELECTED_BANDS]
        print(f"  -> {label_name} 处理完毕，形状: {reduced_data.shape}")
        return reduced_data
    except Exception as e:
        print(f"  ❌ {label_name} 处理失败: {e}")
        return None


if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    # 1. 处理非 PET (.SPE)
    X_non_pet_spe = process_spe_folder(NON_PET_SPE_DIR, "Non-PET-SPE", threshold=0.01)

    # 2. [已跳过] 处理非 PET (.npy)
    print("\n⏩ 跳过 .npy 文件处理...")
    X_non_pet_npy = None
    # X_non_pet_npy = process_npy_files(NPY_DIR) # 注释掉

    # 3. 处理验证集 (假设验证集也是 .SPE)
    X_val_spe = process_spe_folder(VAL_DIR, "Validation-Set", threshold=0.05)

    # --- 保存处理后的数据 ---
    saved_files = []

    # 保存 SPE 来源的非 PET 数据
    if X_non_pet_spe is not None:
        path = os.path.join(SAVE_DIR, 'non_pet_spe_30bands.npy')
        np.save(path, X_non_pet_spe)
        saved_files.append(path)

    # NPY 部分已跳过
    # if X_non_pet_npy is not None:
    #     path = os.path.join(SAVE_DIR, 'non_pet_npy_30bands.npy')
    #     np.save(path, X_non_pet_npy)
    #     saved_files.append(path)

    if X_val_spe is not None:
        path = os.path.join(SAVE_DIR, 'val_data_30bands.npy')
        np.save(path, X_val_spe)
        saved_files.append(path)

    print("\n" + "=" * 50)
    if saved_files:
        print("✅ 数据预处理完成！已保存以下文件：")
        for f in saved_files:
            print(f"  📄 {f}")
        print(f"  (其中 non_pet_spe_30bands.npy 包含 {X_non_pet_spe.shape[0]} 个样本，足够训练使用)")
        print("\n下一步：请运行 train_transformer.py 开始训练！")
    else:
        print("❌ 没有保存任何数据，请检查输入路径是否正确。")