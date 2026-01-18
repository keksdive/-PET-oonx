#include "ChartVisualizer.h"
#include <algorithm> // 用于 std::max_element, std::min_element
#include <iomanip>   // 用于 std::setprecision
#include <sstream>

namespace HSI_Project {

    // --- 构造函数 ---
    ChartVisualizer::ChartVisualizer(std::string windowName, int width, int height)
        : m_windowName(windowName), m_canvasSize(width, height) {
        // 创建一个可调整大小的窗口
        cv::namedWindow(m_windowName, cv::WINDOW_NORMAL);
        cv::resizeWindow(m_windowName, width, height);
    }

    // --- 析构函数 ---
    ChartVisualizer::~ChartVisualizer() {
        cv::destroyWindow(m_windowName);
    }

    // --- 辅助：归一化向量 (用于绘图，不改变原始数据) ---
    std::vector<float> ChartVisualizer::normalizeVector(const std::vector<float>& input) {
        if (input.empty()) return {};
        
        std::vector<float> output = input;
        // 找到最大最小值
        auto minMax = std::minmax_element(output.begin(), output.end());
        float minVal = *minMax.first;
        float maxVal = *minMax.second;
        float range = maxVal - minVal;

        // 避免除以零
        if (range < 1e-6) range = 1.0f;

        for (float& val : output) {
            val = (val - minVal) / range;
        }
        return output;
    }

    // --- 辅助：绘制文字 ---
    void ChartVisualizer::drawInfoText(cv::Mat& img, const std::string& text, cv::Point pos) {
        cv::putText(img, text, pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    // --- 核心功能 1：绘制光谱曲线 (类似示波器) ---
    cv::Mat ChartVisualizer::plotSpectralCurve(const std::vector<float>& spectrumData, const cv::Scalar& color) {
        // 创建黑色背景图
        cv::Mat plot = cv::Mat::zeros(m_plotHeight, m_plotWidth, CV_8UC3);
        
        if (spectrumData.empty()) return plot;

        // 绘制网格背景 (Grid) 方便观察波形特征
        int gridStep = 50;
        for (int i = 0; i < m_plotWidth; i += gridStep) 
            cv::line(plot, cv::Point(i, 0), cv::Point(i, m_plotHeight), cv::Scalar(50, 50, 50), 1);
        for (int i = 0; i < m_plotHeight; i += gridStep) 
            cv::line(plot, cv::Point(0, i), cv::Point(m_plotWidth, i), cv::Scalar(50, 50, 50), 1);

        // 归一化数据以适应窗口高度
        std::vector<float> normData = normalizeVector(spectrumData);
        
        // 绘制波形
        int numPoints = static_cast<int>(normData.size());
        // 计算X轴的步长
        double xStep = (double)m_plotWidth / (numPoints - 1);

        for (int i = 0; i < numPoints - 1; ++i) {
            // Y轴坐标翻转：因为图像坐标系(0,0)在左上角，而我们希望波形底部在下方
            cv::Point p1(static_cast<int>(i * xStep), static_cast<int>(m_plotHeight - normData[i] * (m_plotHeight - 20) - 10));
            cv::Point p2(static_cast<int>((i + 1) * xStep), static_cast<int>(m_plotHeight - normData[i+1] * (m_plotHeight - 20) - 10));
            
            cv::line(plot, p1, p2, color, 2, cv::LINE_AA);
        }

        // 加上标题
        drawInfoText(plot, "Spectral Signature (Normalized)", cv::Point(10, 25));
        return plot;
    }

    // --- 核心功能 2：绘制分类概率柱状图 ---
    cv::Mat ChartVisualizer::plotClassProbabilities(const std::vector<float>& probabilities, 
                                                  const std::vector<std::string>& classLabels) {
        // 这里的宽度稍小一点，高度与光谱图一致
        int width = 400;
        cv::Mat plot = cv::Mat::zeros(m_plotHeight, width, CV_8UC3); // 黑色背景

        if (probabilities.size() != classLabels.size()) {
            drawInfoText(plot, "Error: Dim mismatch", cv::Point(10, 50));
            return plot;
        }

        int numClasses = static_cast<int>(probabilities.size());
        int barHeight = (m_plotHeight - 40) / numClasses; // 每个柱子的高度
        int startY = 30;

        for (int i = 0; i < numClasses; ++i) {
            float prob = probabilities[i]; // 0.0 - 1.0
            
            // 确定柱状条的长度
            int barLen = static_cast<int>(prob * (width - 120)); // 留出空间写文字
            
            // 选个颜色：如果概率大于0.8显示绿色，否则显示黄色/红色
            cv::Scalar barColor = (prob > 0.5) ? cv::Scalar(0, 255, 100) : cv::Scalar(0, 165, 255);

            // 绘制矩形条
            cv::Rect barRect(100, startY + i * barHeight + 5, barLen, barHeight - 10);
            cv::rectangle(plot, barRect, barColor, -1); // -1 表示填充

            // 绘制标签 (PET, Cotton...)
            drawInfoText(plot, classLabels[i], cv::Point(5, startY + i * barHeight + barHeight/2 + 5));

            // 绘制百分比数值
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << (prob * 100.0) << "%";
            drawInfoText(plot, ss.str(), cv::Point(100 + barLen + 5, startY + i * barHeight + barHeight/2 + 5));
        }

        drawInfoText(plot, "Inference Confidence", cv::Point(10, 20));
        return plot;
    }

    // --- 核心功能 3：整合仪表盘 ---
    void ChartVisualizer::updateDashboard(const cv::Mat& rgbImage, 
                                        const std::vector<float>& currentSpectrum,
                                        const std::vector<float>& predictions,
                                        const std::vector<std::string>& labels,
                                        double inferenceTimeMs) {
        
        // 1. 准备画布
        cv::Mat canvas = cv::Mat::zeros(m_canvasSize, CV_8UC3);

        // 2. 处理 RGB 图像区域 (左上角)
        cv::Mat resizedRGB;
        // 保持比例缩放到宽度为 600 左右
        float scale = 600.0f / rgbImage.cols;
        cv::resize(rgbImage, resizedRGB, cv::Size(), scale, scale);
        
        // 限制一下最大高度，防止覆盖下面的图表
        if (resizedRGB.rows > m_canvasSize.height - m_plotHeight - 20) {
            cv::resize(resizedRGB, resizedRGB, cv::Size(600, m_canvasSize.height - m_plotHeight - 20));
        }

        // 将 RGB 贴到画布左上角 (坐标 20, 50)
        cv::Rect rgbROI(20, 50, resizedRGB.cols, resizedRGB.rows);
        resizedRGB.copyTo(canvas(rgbROI));

        // 3. 生成并贴上光谱曲线 (左下角)
        cv::Mat spectralPlot = plotSpectralCurve(currentSpectrum, cv::Scalar(0, 255, 255)); // 黄色曲线
        // 位置：在 RGB 下方
        int plotY = m_canvasSize.height - m_plotHeight - 20;
        cv::Rect specROI(20, plotY, spectralPlot.cols, spectralPlot.rows);
        spectralPlot.copyTo(canvas(specROI));

        // 4. 生成并贴上概率图 (右侧)
        cv::Mat probPlot = plotClassProbabilities(predictions, labels);
        // 位置：RGB 右侧
        cv::Rect probROI(640, 50, probPlot.cols, probPlot.rows);
        probPlot.copyTo(canvas(probROI));

        // 5. 绘制全局标题和状态栏
        cv::rectangle(canvas, cv::Rect(0, 0, m_canvasSize.width, 40), cv::Scalar(30, 30, 30), -1); // 顶部标题栏背景
        cv::putText(canvas, "HSI-RGB Multi-modal Fusion Sorting System (Demo)", cv::Point(20, 28), 
                    cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(255, 255, 255), 1);

        // 显示 FPS / 耗时
        std::stringstream ss;
        ss << "Inference: " << inferenceTimeMs << " ms  |  Device: GPU (CUDA/TensorRT)"; 
        // 这里的 Device 文字你可以根据实际情况改成 CPU (ONNX Runtime) 或其他
        
        cv::putText(canvas, ss.str(), cv::Point(20, m_canvasSize.height - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 150), 1);

        // 6. 显示最终画面
        cv::imshow(m_windowName, canvas);
        cv::waitKey(1); // 必须有这一句，否则窗口不刷新
    }

} // namespace HSI_Project