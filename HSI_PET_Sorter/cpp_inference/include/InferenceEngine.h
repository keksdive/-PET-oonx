#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// ONNX Runtime 核心头文件
#include <onnxruntime_cxx_api.h>

namespace HSI_Project {

    class InferenceEngine {
    public:
        /**
         * @brief 构造函数
         * @param modelPath ONNX 模型文件的路径 (例如 "assets/models/spectral_net.onnx")
         */
        InferenceEngine(const std::string& modelPath);
        ~InferenceEngine();

        /**
         * @brief 执行推理 (通用版，适用于 1D 数据)
         * 适用于：
         * 1. 光谱数据 (1 x Bands)
         * 2. 颜色特征向量 (1 x 48)
         * * @param inputData 归一化后的输入向量
         * @return std::vector<float> 每个类别的概率值 (经过 Softmax)
         */
        std::vector<float> runInference(const std::vector<float>& inputData);

    private:
        // --- ONNX Runtime 核心组件 ---
        Ort::Env m_env;
        Ort::SessionOptions m_sessionOptions;
        std::unique_ptr<Ort::Session> m_session;
        Ort::AllocatorWithDefaultOptions m_allocator;

        // --- 模型元数据 ---
        std::vector<const char*> m_inputNodeNames;
        std::vector<const char*> m_outputNodeNames;
        std::vector<int64_t> m_inputShape;  // 模型期待的输入形状，例如 [1, 200]
        
        // 字符串存储 (用于保持指针有效性)
        std::vector<std::string> m_inputNameStrings;
        std::vector<std::string> m_outputNameStrings;

        /**
         * @brief 辅助函数：Softmax 归一化
         * 将模型输出的 Logits 转换为概率 (0.0 - 1.0)
         */
        std::vector<float> softmax(const std::vector<float>& logits);
    };

} // namespace HSI_Project