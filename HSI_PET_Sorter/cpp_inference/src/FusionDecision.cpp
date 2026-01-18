#include "FusionDecision.h"

namespace HSI_Project {

    FusionDecision::FusionDecision(int numClasses) : m_numClasses(numClasses) {
        // 初始化：默认所有模型权重为 0.5 (平局)
        m_hsiClassWeights.assign(numClasses, 0.5f);
        m_rgbClassWeights.assign(numClasses, 0.5f);
    }

    FusionDecision::~FusionDecision() {}

    // ★ 关键实现：权重计算与归一化
    void FusionDecision::setFMeasureWeights(const std::vector<float>& hsiFMeasure, 
                                            const std::vector<float>& rgbFMeasure) {
        if (hsiFMeasure.size() != m_numClasses || rgbFMeasure.size() != m_numClasses) {
            std::cerr << "[Fusion Error] F-Measure vector size mismatch! Expected " << m_numClasses << std::endl;
            return;
        }

        std::cout << "[Fusion] Updating Weights based on Validation Set F-Measures..." << std::endl;

        for (int i = 0; i < m_numClasses; ++i) {
            float f_hsi = hsiFMeasure[i];
            float f_rgb = rgbFMeasure[i];
            float sum = f_hsi + f_rgb;

            // 归一化处理：使得 w_hsi + w_rgb = 1
            if (sum < 1e-6) {
                // 如果两个模型在该类别上 F值都是0 (极其罕见)，则五五开
                m_hsiClassWeights[i] = 0.5f;
                m_rgbClassWeights[i] = 0.5f;
            } else {
                m_hsiClassWeights[i] = f_hsi / sum;
                m_rgbClassWeights[i] = f_rgb / sum;
            }

            // 打印调试信息，让你看到每个类别的“偏心”程度
            std::cout << "  Class " << i << " Weights -> HSI: " << m_hsiClassWeights[i] 
                      << " | RGB: " << m_rgbClassWeights[i] << std::endl;
        }
    }

    // ★ 关键实现：最终得分计算
    std::vector<float> FusionDecision::computeFusion(const std::vector<float>& hsiProbs, 
                                                     const std::vector<float>& rgbProbs) {
        // 1. 安全检查
        if (hsiProbs.size() != m_numClasses || rgbProbs.size() != m_numClasses) {
            // 如果输入维度不对，直接返回 HSI 结果作为降级方案
            return hsiProbs.empty() ? rgbProbs : hsiProbs;
        }

        std::vector<float> finalProbs(m_numClasses);

        // 2. 逐类别加权 (Class-wise Weighting)
        // Def: Final_j = HSI_Prob_j * HSI_Weight_j + RGB_Prob_j * RGB_Weight_j
        for (int i = 0; i < m_numClasses; ++i) {
            finalProbs[i] = (hsiProbs[i] * m_hsiClassWeights[i]) + 
                            (rgbProbs[i] * m_rgbClassWeights[i]);
        }

        // 3. 重新归一化 (Renormalization)
        // 因为每个类别的权重不同，简单的加权和可能导致最终向量之和不为 1
        // 为了方便后续 Defuzzification (比较大小) 和可视化，建议重新归一化
        normalizeProbabilities(finalProbs);

        return finalProbs;
    }

    void FusionDecision::normalizeProbabilities(std::vector<float>& probs) {
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (sum > 1e-6) {
            for (auto& val : probs) {
                val /= sum;
            }
        }
    }

    // 最终决策 (Defuzzification)：选择最大值
    int FusionDecision::getFinalClass(const std::vector<float>& finalProbs) {
        if (finalProbs.empty()) return -1;
        auto maxIt = std::max_element(finalProbs.begin(), finalProbs.end());
        return static_cast<int>(std::distance(finalProbs.begin(), maxIt));
    }

} // namespace HSI_Project