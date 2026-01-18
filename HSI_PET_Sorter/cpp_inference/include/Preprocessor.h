#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

namespace HSI_Project {

    class Preprocessor {
    public:
        Preprocessor();
        ~Preprocessor();

        // ==========================================
        // 🌟 新增核心功能：HSI 数据分流
        // ==========================================

        /**
         * @brief HSI 模型专用：提取特定波段
         * 根据 DRL 训练出的波段索引，从全波段数据中提取特征向量。
         * * @param rawCube 原始高光谱数据 (通常为多通道 cv::Mat)
         * @param x 当前处理像素的 X 坐标
         * @param y 当前处理像素的 Y 坐标
         * @param selectedBands 需要提取的波段索引列表 (如 {10, 25, 40...})
         * @return cv::Mat 返回 1xN 的浮点型行向量 (CV_32FC1)，可直接输入 HSI 推理引擎
         */
        cv::Mat extractSelectedBands(const cv::Mat& rawCube, int x, int y, const std::vector<int>& selectedBands);

        // ==========================================
        // 🌟 新增核心功能：RGB 数据分流
        // ==========================================

        /**
         * @brief RGB 模型专用：合成伪彩色图像并适配尺寸
         * 提取指定的三个波段合成 RGB 图像，并缩放到模型所需的输入尺寸 (如 224x224)。
         * * @param rawCube 原始高光谱数据
         * @param x 当前中心像素 X 坐标 (用于截取 Patch，如果是全图处理则忽略)
         * @param y 当前中心像素 Y 坐标
         * @param rIdx 红通道波段索引
         * @param gIdx 绿通道波段索引
         * @param bIdx 蓝通道波段索引
         * @param targetSize 模型输入尺寸，默认 224x224
         * @return cv::Mat 返回合成并缩放后的 3 通道彩色图像 (CV_8UC3 或 CV_32FC3)
         */
        cv::Mat generatePseudoRGB(const cv::Mat& rawCube, int x, int y,
            int rIdx, int gIdx, int bIdx,
            cv::Size targetSize = cv::Size(224, 224));


        // ==========================================
        // 旧有功能 (保留以兼容旧逻辑)
        // ==========================================

        /**
         * @brief 提取颜色矩特征
         */
        static std::vector<float> extractColorMoments(const cv::Mat& img);

        /**
         * @brief 光谱黑白校正
         */
        static std::vector<float> calibrateSpectrum(const std::vector<float>& raw,
            const std::vector<float>& darkRef,
            const std::vector<float>& whiteRef);
    };

} // namespace HSI_Project