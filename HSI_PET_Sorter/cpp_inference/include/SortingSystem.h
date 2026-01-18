#ifndef SORTING_SYSTEM_H
#define SORTING_SYSTEM_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>

// 引入你现有的模块头文件
#include "SpeLoader.h"
#include "Preprocessor.h"
#include "InferenceEngine.h"
#include "FusionDecision.h"
#include "ChartVisualizer.h"

// ==========================================
// ⚙️ 全局配置区域 (Configuration)
// 这里是你在 Python 训练完后需要修改参数的地方
// ==========================================
namespace Config {
    // 1. 模型路径
    const std::string HSI_MODEL_PATH = "models/hsi_model.onnx";
    const std::string RGB_MODEL_PATH = "models/rgb_model.onnx";

    // 2. DRL 选出的 30 个波段索引 (请从 Python training.py 输出结果复制粘贴到这里)
    // 示例: {10, 25, 40, ..., 190}
    const std::vector<int> SELECTED_HSI_BANDS = {
        10, 25, 33, 45, 50, 60, 72, 80, 90, 100,
        110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
        // ... 确保这里凑够你训练时的数量 (如30个)
    };

    // 3. 用于合成伪 RGB 的波段索引 (对应 R, G, B)
    // 请确保这三个索引包含在原始数据的波段范围内
    const int RGB_BAND_R = 100; // 红光波段索引
    const int RGB_BAND_G = 60;  // 绿光波段索引
    const int RGB_BAND_B = 20;  // 蓝光波段索引

    // 4. 软投票权重配置
    const float WEIGHT_HSI = 0.8f; // HSI 权重 (通常较高)
    const float WEIGHT_RGB = 0.2f; // RGB 权重
    const float DECISION_THRESHOLD = 0.5f; // 最终判定 PET 的阈值
}

// ==========================================
// 🧠 核心系统类 (System Controller)
// ==========================================
class SortingSystem {
public:
    SortingSystem();
    ~SortingSystem();

    // 初始化系统：加载两个模型
    bool initialize();

    // 运行处理流程：传入光谱文件路径
    void run(const std::string& speFilePath);

private:
    // --- 内部处理函数 ---
    // 单帧处理逻辑：输入原始数据，输出最终结果
    void processFrame(const cv::Mat& rawCube);

private:
    // --- 模块实例 ---
    SpeLoader       m_loader;
    Preprocessor    m_preprocessor;

    // 关键修改：双推理引擎
    InferenceEngine m_hsiEngine; // 负责光谱推理
    InferenceEngine m_rgbEngine; // 负责图像推理

    FusionDecision  m_decisionMaker;
    ChartVisualizer m_visualizer;

    // --- 运行时状态 ---
    bool m_isInitialized = false;
};

#endif // SORTING_SYSTEM_H#pragma once
