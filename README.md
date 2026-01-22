现有代码满足要求的程度
训练独立的分类模型: 满足。train\_transformer.py 构建了一个包含 1D-CNN 和 Transformer Encoder 的光谱分类器。

输出为 ONNX 格式: 满足。代码中已有 tf2onnx 转换逻辑，且 export\_onnx.py 专门负责此项工作。

像素点级材质判别: 满足。模型输入为 (Batch, 30) 的光谱向量，推理时是对图像进行像素级遍历或批处理判别。

概率热力图: 满足。inference\_demo.py 中已经实现了使用 jet 颜色映射生成检测热力图。

# -PET-oonx

一个用于高光谱（HSI）像素级分类的仓库，包含模型训练、ONNX 导出和基于 ONNX Runtime 的推理示例。项目以识别 PET（或其它材料）为目标，支持从 .spe/.hdr 格式影像读取、辐射校准、像素级归一化和批量推理。

## 目录结构（重要文件）

- `train_transformer.py`：训练 1D-CNN + Transformer 的分类模型。
- `export_onnx.py`：将训练好的模型导出为 ONNX（供推理使用）。
- `inference_demo.py`：主推理脚本，加载 ONNX 模型并对目录下的 `.spe` 文件进行像素级推理与可视化。
- `best_bands_config.json`：（运行训练后生成）记录选定的波段索引，推理脚本会读取此文件以确定输入通道。
- `WHITE_REF` / `DARK_REF`：白场与暗场校准文件（`.spe`），用于从原始 DN 值计算反射率。
- 其它：`q_network.py`、`agent.py`、`reward_utils.py` 等为训练/算法模块或实验代码。

## 快速上手

1. 安装依赖（建议在虚拟环境中）：

```powershell
pip install -r requirements.txt
```

2. 准备文件：

- 将 ONNX 模型放到某个路径（示例中使用绝对路径在 `inference_demo.py` 顶部配置）。
- 准备待推理的 `.spe` 文件，放在 `INPUT_DIR` 指定的目录。
- 确保有白场/暗场校准文件（`.spe`）用于反射率计算。
- 确保 `best_bands_config.json` 存在并包含键 `selected_bands`（值为波段索引列表）。训练脚本 `train_transformer.py` 会生成该文件（或手动创建）。

示例 `best_bands_config.json`：

```json
{
	"selected_bands": [10, 12, 25, 30, 45, 60, 72, 89, 101, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405, 420]
}
```

3. 在 `inference_demo.py` 中检查并更新配置（脚本顶部常量）:

- `MODEL_PATH`：ONNX 模型路径
- `CONFIG_PATH`：`best_bands_config.json` 路径
- `INPUT_DIR`：待处理影像目录
- `OUTPUT_DIR`：结果（热力图、npy）的保存目录
- `WHITE_REF` / `DARK_REF`：白场/暗场校准文件
- `BRIGHTNESS_THRESHOLD` / `CONFIDENCE_THRESHOLD`：可调的阈值

4. 运行推理脚本：

```powershell
python inference_demo.py
```

处理结果会输出到 `OUTPUT_DIR`，包含每张影像的 `_result.png`（热力图）和 `_pred.npy`（浮点掩码）。脚本运行时会打印每张图像的推理时间和保存状态。

## 关键实现细节

- 波段选择：`inference_demo.py` 会读取 `best_bands_config.json` 中的 `selected_bands`，并以此索引从原始 HSI 中选取波段。
- 校准流程：脚本通过读取 `WHITE_REF` 和 `DARK_REF` 的平均光谱来做逐像素反射率计算：
	reflectance = (raw - dark) / (white - dark)
- 归一化：对每个像素独立做 min-max 归一化，降低光照不均匀影响。
- 推理：使用 ONNX Runtime 加载 ONNX 模型，对有效像素批量推理，然后基于置信度阈值生成二值预测图。

## 训练与导出

- 使用 `train_transformer.py` 训练模型（参看脚本内部注释）。
- 使用 `export_onnx.py` 将训练得到的模型导出为 ONNX。导出时注意模型输入维度必须与 `len(selected_bands)` 中的波段数一致。

## 常见问题与排查

- 无法加载 ONNX：检查 `MODEL_PATH` 是否正确；确认 ONNX 的输入维度与 `len(selected_bands)` 匹配。
- 找不到 `best_bands_config.json`：在根目录运行训练导出脚本，或手动创建该 JSON。脚本会在启动时退出并打印错误信息。
- .spe/.hdr 读取错误：脚本会尝试修复 `.hdr` 的 `byte order` 字段，如果仍然失败，确认文件未损坏且 ENVI 格式正确。
- 出现 NaN/Inf：校准差分 (white - dark) 出现 0 会被脚本替换为小的常数以避免除 0；若反射率仍异常，请检查校准文件是否合适。

## 文件说明

- `inference_demo.py`：推理入口，建议根据环境修改顶部配置常量。
- `train_transformer.py`：训练脚本（包含数据加载、模型定义、训练循环）。
- `export_onnx.py`：模型转换/导出工具。
- `best_bands_config.json`：自动/手动生成的波段索引配置，用于推理时选择通道。

## 许可证

请根据项目实际选择合适许可证（当前仓库未在此处声明）。

---

如果你希望我把 `inference_demo.py` 的配置改为从环境变量或命令行参数读取（更灵活），我可以为你修改脚本并添加示例命令。欢迎告诉我你的偏好。

