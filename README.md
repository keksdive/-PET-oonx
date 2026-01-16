现有代码满足要求的程度
训练独立的分类模型: 满足。train_transformer.py 构建了一个包含 1D-CNN 和 Transformer Encoder 的光谱分类器。

输出为 ONNX 格式: 满足。代码中已有 tf2onnx 转换逻辑，且 export_onnx.py 专门负责此项工作。

像素点级材质判别: 满足。模型输入为 (Batch, 30) 的光谱向量，推理时是对图像进行像素级遍历或批处理判别。

概率热力图: 满足。inference_demo.py 中已经实现了使用 jet 颜色映射生成检测热力图。
