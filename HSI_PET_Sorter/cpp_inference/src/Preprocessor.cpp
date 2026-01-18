#include "Preprocessor.h"

namespace HSI_Project {

    Preprocessor::Preprocessor() {}
    Preprocessor::~Preprocessor() {}

    // --- 核心算法：4x4 网格颜色矩提取 ---
    std::vector<float> Preprocessor::extractColorMoments(const cv::Mat& img) {
        std::vector<float> features;
        features.reserve(48); // 预分配内存，提高效率

        if (img.empty()) {
            std::cerr << "[Error] Input image is empty in extractColorMoments!" << std::endl;
            // 返回全0向量防止崩溃
            return std::vector<float>(48, 0.0f);
        }

        // 1. 确保是 RGB 格式 (OpenCV 默认是 BGR，这很重要！)
        // 如果你的 ONNX 模型是用 RGB 训练的，这里必须转。
        // 如果是用 BGR 训练的，请注释掉这一行。通常 PyTorch 读图是 RGB。
        cv::Mat rgbImg;
        cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);

        // 2. 计算网格尺寸
        int rows = rgbImg.rows;
        int cols = rgbImg.cols;
        
        // 划分为 4x4 = 16 个区域
        int gridRows = 4;
        int gridCols = 4;
        
        int cellHeight = rows / gridRows;
        int cellWidth = cols / gridCols;

        // 3. 遍历 16 个格子
        for (int r = 0; r < gridRows; ++r) {
            for (int c = 0; c < gridCols; ++c) {
                // 定义 ROI (感兴趣区域)
                int x = c * cellWidth;
                int y = r * cellHeight;
                
                // 边界检查，防止除不尽导致的越界
                int w = (c == gridCols - 1) ? (cols - x) : cellWidth;
                int h = (r == gridRows - 1) ? (rows - y) : cellHeight;

                cv::Rect roiRect(x, y, w, h);
                cv::Mat roi = rgbImg(roiRect);

                // 4. 计算该区域的均值 (Mean)
                // cv::mean 返回 4个通道的值 (Scalar)，我们要前3个
                cv::Scalar avgPixel = cv::mean(roi);

                // 5. 归一化并存入向量
                // 神经网络通常喜欢 0.0 - 1.0 的输入，而不是 0 - 255
                features.push_back(static_cast<float>(avgPixel[0] / 255.0f)); // R
                features.push_back(static_cast<float>(avgPixel[1] / 255.0f)); // G
                features.push_back(static_cast<float>(avgPixel[2] / 255.0f)); // B
            }
        }

        // 此时 features.size() 应该严格等于 48
        return features;
    }

    // --- 辅助：光谱黑白校正 ---
    std::vector<float> Preprocessor::calibrateSpectrum(const std::vector<float>& raw, 
                                                       const std::vector<float>& darkRef, 
                                                       const std::vector<float>& whiteRef) {
        if (raw.size() != darkRef.size() || raw.size() != whiteRef.size()) {
            std::cerr << "[Warning] Calibration dimension mismatch!" << std::endl;
            return raw;
        }

        std::vector<float> calibrated(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            float denominator = whiteRef[i] - darkRef[i];
            // 防止分母为0
            if (std::abs(denominator) < 1e-5) denominator = 1.0f;
            
            float val = (raw[i] - darkRef[i]) / denominator;
            
            // 截断到 0.0 - 1.0 之间
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            
            calibrated[i] = val;
        }
        return calibrated;
    }

} // namespace HSI_Project