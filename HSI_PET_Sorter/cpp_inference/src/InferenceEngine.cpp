#include "InferenceEngine.h"
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace HSI_Project {

    // 构造函数：加载模型并获取输入输出信息
    InferenceEngine::InferenceEngine(const std::string& modelPath)
        : m_env(ORT_LOGGING_LEVEL_WARNING, "HSI_Inference"), 
          m_sessionOptions() 
    {
        // 1. 设置会话选项 (可开启图优化)
        m_sessionOptions.SetIntraOpNumThreads(1);
        m_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // 2. 加载模型 (Windows 下路径需转为宽字符)
        // 将 std::string 转为 std::wstring (Windows 专用)
        std::wstring wModelPath(modelPath.begin(), modelPath.end());

        try {
            m_session = std::make_unique<Ort::Session>(m_env, wModelPath.c_str(), m_sessionOptions);
        } catch (const Ort::Exception& e) {
            std::cerr << "[Error] Failed to load ONNX model: " << modelPath << "\n";
            std::cerr << e.what() << std::endl;
            throw;
        }

        // 3. 自动解析模型的输入输出节点名称和形状
        size_t numInputNodes = m_session->GetInputCount();
        size_t numOutputNodes = m_session->GetOutputCount();

        // 获取输入节点信息
        for (size_t i = 0; i < numInputNodes; i++) {
            // 获取名称
            auto inputName = m_session->GetInputNameAllocated(i, m_allocator);
            m_inputNameStrings.push_back(inputName.get()); // 保存字符串副本
            m_inputNodeNames.push_back(m_inputNameStrings.back().c_str());

            // 获取形状
            auto typeInfo = m_session->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            m_inputShape = tensorInfo.GetShape();
            
            // 修正动态维度 (如果是 -1，设为 1)
            if (m_inputShape[0] == -1) m_inputShape[0] = 1; 
        }

        // 获取输出节点信息
        for (size_t i = 0; i < numOutputNodes; i++) {
            auto outputName = m_session->GetOutputNameAllocated(i, m_allocator);
            m_outputNameStrings.push_back(outputName.get());
            m_outputNodeNames.push_back(m_outputNameStrings.back().c_str());
        }

        std::cout << "[Info] Model loaded: " << modelPath << std::endl;
        std::cout << "[Info] Input Shape: [" << m_inputShape[0] << ", " << m_inputShape[1] << "]" << std::endl;
    }

    InferenceEngine::~InferenceEngine() {
        // Unique_ptr 会自动释放 Session
    }

    // 执行推理
    std::vector<float> InferenceEngine::runInference(const std::vector<float>& inputData) {
        // 1. 检查数据长度是否匹配
        // m_inputShape[1] 是特征维度 (例如 200 或 48)
        if (inputData.size() != static_cast<size_t>(m_inputShape[1])) {
            std::cerr << "[Warning] Input data size mismatch! Expected " << m_inputShape[1] 
                      << ", got " << inputData.size() << std::endl;
            return {};
        }

        // 2. 创建输入 Tensor
        // ONNX Runtime 需要非 const 的 void* 指针，所以这里要 const_cast 或者拷贝
        // 这是一个内存视图，不涉及深拷贝
        std::vector<float> inputCopy = inputData; // 拷贝一份以确保内存安全
        
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, 
            inputCopy.data(),           // 数据指针
            inputCopy.size(),           // 数据总长度
            m_inputShape.data(),        // 形状数组指针 ([1, 200])
            m_inputShape.size()         // 形状数组长度 (2)
        );

        // 3. 运行推理 (Run)
        auto outputTensors = m_session->Run(
            Ort::RunOptions{nullptr},
            m_inputNodeNames.data(),    // 输入节点名数组
            &inputTensor,               // 输入 Tensor 数组
            1,                          // 输入个数
            m_outputNodeNames.data(),   // 输出节点名数组
            1                           // 输出个数
        );

        // 4. 解析输出
        float* floatArr = outputTensors[0].GetTensorMutableData<float>();
        size_t outputSize = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> outputLogits(floatArr, floatArr + outputSize);

        // 5. Softmax 处理 (Logits -> Probabilities)
        return softmax(outputLogits);
    }

    // 辅助：Softmax
    std::vector<float> InferenceEngine::softmax(const std::vector<float>& logits) {
        if (logits.empty()) return {};

        std::vector<float> probs = logits;
        
        // 找到最大值 (为了数值稳定性，防止 exp 溢出)
        float maxVal = *std::max_element(probs.begin(), probs.end());
        
        float sum = 0.0f;
        for (auto& val : probs) {
            val = std::exp(val - maxVal);
            sum += val;
        }
        
        for (auto& val : probs) {
            val /= sum;
        }
        
        return probs;
    }

} // namespace HSI_Project