#pragma once

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace HSI_Project {

    class FusionDecision {
    public:
        /**
         * @brief 构造函数
         * @param numClasses 类别总数 (例如 3: PET, Cotton, Mix)
         */
        FusionDecision(int numClasses);
        ~FusionDecision();

        /**
         * @brief ★ 核心功能：设置基于验证集 F-measure 的权重
         * 这是你论文要求的关键步骤。
         * 系统会自动归一化：w_hsi[i] + w_rgb[i] = 1.0
         * * @param hsiFMeasure 在验证集上测得的 HSI 模型对各类的 F-Measure
         * @param rgbFMeasure 在验证集上测得的 RGB 模型对各类的 F-Measure
         */
        void setFMeasureWeights(const std::vector<float>& hsiFMeasure, 
                                const std::vector<float>& rgbFMeasure);

        /**
         * @brief 执行基于类别的软投票 (Class-Specific Soft Voting)
         * Formula: P_final[j] = P_hsi[j] * W_hsi[j] + P_rgb[j] * W_rgb[j]
         */
        std::vector<float> computeFusion(const std::vector<float>& hsiProbs, 
                                         const std::vector<float>& rgbProbs);

        /**
         * @brief 最终决策 (Defuzzification)
         * 获取概率最大的类别索引
         */
        int getFinalClass(const std::vector<float>& finalProbs);

    private:
        int m_numClasses;

        // 存储每个类别的专属权重
        // 例如: m_hsiClassWeights[0] 是 HSI 对类别 0 (PET) 的权重
        std::vector<float> m_hsiClassWeights;
        std::vector<float> m_rgbClassWeights;

        // 内部辅助：重新归一化概率向量，使其和为1
        void normalizeProbabilities(std::vector<float>& probs);
    };

} // namespace HSI_Project