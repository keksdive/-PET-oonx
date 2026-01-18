#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>

// 命名空间管理，防止与后续可能引入的库冲突
namespace HSI_Project {

    /**
     * @brief 可视化图表生成器
     * 用于将高光谱数据、推理概率和原始图像整合成一个可视化仪表盘
     */
    class ChartVisualizer {
    public:
        /**
         * @brief 构造函数
         * @param windowName 显示窗口的名称
         * @param width 仪表盘总宽度 (默认 1280)
         * @param height 仪表盘总高度 (默认 720)
         */
        ChartVisualizer(std::string windowName = "HSI Sorting Dashboard", int width = 1280, int height = 720);

        /**
         * @brief 析构函数
         */
        ~ChartVisualizer();

        /**
         * @brief 绘制光谱曲线图 (Line Chart)
         * 将高光谱传感器采集的一维向量转换为波形图
         * * @param spectrumData 输入的光谱强度向量 (例如 200-400个波段的数据)
         * @param color 曲线颜色
         * @return cv::Mat 返回绘制好的光谱图图像
         */
        cv::Mat plotSpectralCurve(const std::vector<float>& spectrumData, 
                                  const cv::Scalar& color = cv::Scalar(0, 255, 0));

        /**
         * @brief 绘制分类概率柱状图 (Bar Chart)
         * 用于直观显示 AI 认为当前物体属于 PET、棉还是混纺的概率
         * * @param probabilities 模型输出的 Softmax 概率向量
         * @param classLabels 对应的类别名称列表 (如 {"PET", "Cotton", "Mix"})
         * @return cv::Mat 返回绘制好的柱状图图像
         */
        cv::Mat plotClassProbabilities(const std::vector<float>& probabilities, 
                                       const std::vector<std::string>& classLabels);

        /**
         * @brief 核心更新函数：刷新整个仪表盘
         * 将 RGB 图像、光谱图、概率图拼合在一起并显示
         * * @param rgbImage 工业相机采集的原始 RGB 图像
         * @param currentSpectrum 当前像素或区域的光谱数据
         * @param predictions 模型预测的概率向量
         * @param labels 类别标签
         * @param inferenceTimeMs 推理耗时 (用于显示 FPS)
         */
        void updateDashboard(const cv::Mat& rgbImage, 
                             const std::vector<float>& currentSpectrum,
                             const std::vector<float>& predictions,
                             const std::vector<std::string>& labels,
                             double inferenceTimeMs);

    private:
        std::string m_windowName;
        cv::Size m_canvasSize;
        
        // 子图表的尺寸配置
        const int m_plotHeight = 300;
        const int m_plotWidth = 500;

        /**
         * @brief 辅助函数：标准化数据到 0-1 范围，方便绘图
         */
        std::vector<float> normalizeVector(const std::vector<float>& input);

        /**
         * @brief 辅助函数：在图像上绘制文字信息
         */
        void drawInfoText(cv::Mat& img, const std::string& text, cv::Point pos);
    };

} // namespace HSI_Project