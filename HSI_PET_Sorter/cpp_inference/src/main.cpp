#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

// 引入各模块头文件
#include "SpeLoader.h"
#include "Preprocessor.h"
#include "InferenceEngine.h"
#include "FusionDecision.h"
#include "ChartVisualizer.h"

// ==========================================
// ⚙️ 全局配置参数 (Configuration)
// 这里填入你从 Python 训练得到的关键参数
// ==========================================
namespace Config {
    // 1. 模型文件路径
    const std::string MODEL_PATH_HSI = "hsi_model.onnx";
    const std::string MODEL_PATH_RGB = "rgb_model.onnx";

    // 2. DRL 选出的 30 个波段索引 (示例数据，请替换为你 training.py 的真实输出)
    const std::vector<int> SELECTED_BANDS = {
        10, 25, 33, 45, 52, 60, 68, 75, 82, 90,
        101, 110, 115, 122, 130, 138, 145, 150, 160, 172,
        180, 185, 190, 195, 200, 205, 210, 215, 220, 225
    };

    // 3. 伪 RGB 合成波段 (Red, Green, Blue 对应的波段索引)
    // 请确保这些索引与 Python 训练 RGB 模型时一致
    const int BAND_R = 100;
    const int BAND_G = 60;
    const int BAND_B = 20;

    // 4. 融合权重与阈值
    const float WEIGHT_HSI = 0.7f;
    const float WEIGHT_RGB = 0.3f;
    const float CONFIDENCE_THRESHOLD = 0.6f; // 超过 0.6 判定为 PET
}

int main(int argc, char** argv) {
    // --- 1. 实例化各功能模块 ---
    std::cout << "[System] Initializing modules..." << std::endl;

    SpeLoader       loader;
    Preprocessor    preprocessor;
    FusionDecision  decisionMaker;
    ChartVisualizer visualizer;

    // 关键：实例化双推理引擎
    InferenceEngine hsiEngine;
    InferenceEngine rgbEngine;

    // --- 2. 加载模型 (Load Models) ---
    std::cout << "[System] Loading AI models..." << std::endl;

    // 注意：这里可能需要你在 InferenceEngine 中实现 setModelType 或类似的配置
    // 以便引擎知道如何处理输入 (例如 RGB 需要 Resize，HSI 不需要)
    if (!hsiEngine.loadModel(Config::MODEL_PATH_HSI)) {
        std::cerr << "❌ Failed to load HSI model!" << std::endl;
        return -1;
    }
    if (!rgbEngine.loadModel(Config::MODEL_PATH_RGB)) {
        std::cerr << "❌ Failed to load RGB model!" << std::endl;
        return -1;
    }

    // --- 3. 加载测试数据 (Load Data) ---
    std::string testFile = "test_data.spe"; // 实际使用时可从 argv[1] 获取
    if (argc > 1) testFile = argv[1];

    std::cout << "[System] Loading spectral data: " << testFile << std::endl;
    if (!loader.loadSpeFile(testFile)) {
        std::cerr << "❌ Failed to load SPE file." << std::endl;
        return -1;
    }

    // 获取原始高光谱数据 (假设 Shape: H x W x Bands)
    cv::Mat rawCube = loader.getData();
    if (rawCube.empty()) {
        std::cerr << "❌ Data is empty!" << std::endl;
        return -1;
    }

    std::cout << "[System] Data loaded. Shape: " << rawCube.size << " Channels: " << rawCube.channels() << std::endl;

    // --- 4. 循环处理 (Processing Loop) ---
    // 这里演示逐个像素处理，实际工程中建议 Batch 处理或逐行处理

    int height = rawCube.rows;
    int width = rawCube.cols;
    int petCount = 0;

    // 为了演示方便，我们只取中心区域的一个像素进行模拟，或者遍历全图
    // 实际分选机中，这里会是一个 while(camera.capture()) 的死循环
    std::cout << "[System] Starting inference loop..." << std::endl;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            // 4.1 获取当前像素/区域的原始全波段数据
            // 注意：OpenCV 的 Mat 访问需要根据你的数据类型调整，这里假设是 float
            // 取出一个 1x1xBands 的向量 (或者 ROI 区域)
            cv::Mat rawPixel = rawCube.at<cv::Vec<float, 200>>(y, x); // 假设200个波段，仅作示意，需动态处理
            // 如果你的 Preprocessor 接收的是整图和坐标，则传入 x, y

            // --- 4.2 数据分流与预处理 (Preprocessing) ---

            // A. 生成 HSI 输入 (只取 30 个波段)
            // 调用我们在 Preprocessor 中新增的函数
            cv::Mat inputHSI = preprocessor.extractSelectedBands(rawCube, x, y, Config::SELECTED_BANDS);

            // B. 生成 RGB 输入 (合成伪彩色图像)
            // 调用我们在 Preprocessor 中新增的函数 (注意：RGB模型通常需要比如 224x224 的邻域图像)
            // 如果是单像素分类，这里可能是 1x1 的 3通道值，取决于你 RGB 模型的训练方式
            cv::Mat inputRGB = preprocessor.generatePseudoRGB(rawCube, x, y, Config::BAND_R, Config::BAND_G, Config::BAND_B);

            // --- 4.3 双流推理 (Dual Inference) ---

            // HSI 推理 -> 得到概率 (0.0~1.0)
            cv::Mat resHSI = hsiEngine.runInference(inputHSI);
            float probHSI = resHSI.at<float>(0);

            // RGB 推理 -> 得到概率 (0.0~1.0)
            cv::Mat resRGB = rgbEngine.runInference(inputRGB);
            float probRGB = resRGB.at<float>(0);

            // --- 4.4 融合决策 (Decision Fusion) ---

            float finalScore = decisionMaker.softVoting(
                probHSI, probRGB,
                Config::WEIGHT_HSI, Config::WEIGHT_RGB
            );

            bool isPET = (finalScore > Config::CONFIDENCE_THRESHOLD);

            // --- 4.5 结果执行 (Action) ---
            if (isPET) {
                petCount++;
                // 在这里调用气阀控制函数
                // AirNozzle::blow(x, y);
            }

            // 可视化 (可选，仅调试时开启，否则拖慢速度)
            if (x % 100 == 0 && y % 100 == 0) {
                visualizer.updateChart(probHSI, probRGB, finalScore);
                std::cout << "Pos(" << x << "," << y << ") -> Score: " << finalScore
                    << " (HSI:" << probHSI << ", RGB:" << probRGB << ")" << std::endl;
            }
        }
    }

    std::cout << "[System] Processing finished. Total PET detected: " << petCount << std::endl;

    return 0;
}