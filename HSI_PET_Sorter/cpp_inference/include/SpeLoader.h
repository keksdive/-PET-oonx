#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

namespace HSI_Project {

    /**
     * @brief 高光谱数据加载器 (支持 ENVI 标准格式)
     * 针对数据类型 2 (int16) 和 12 (uint16) 进行了优化
     */
    class SpeLoader {
    public:
        SpeLoader();
        ~SpeLoader();

        /**
         * @brief 加载高光谱数据文件 (.hdr)
         * @param filePath .hdr 文件的完整路径
         * @return bool 是否加载成功
         */
        bool load(const std::string& filePath);

        /**
         * @brief 显式清空当前加载的数据，释放内存
         */
        void close();

        /**
         * @brief 获取指定像素点的光谱曲线
         * @param x 图像列坐标 (Sample)
         * @param y 图像行坐标 (Line)
         * @return std::vector<float> 提取的光谱向量
         */
        std::vector<float> getSpectrum(int x, int y) const;

        // --- 属性获取接口 ---
        int getWidth() const { return m_samples; }
        int getHeight() const { return m_lines; }
        int getBands() const { return m_bands; }
        int getDataType() const { return m_dataType; }
        std::string getInterleave() const { return m_interleave; }

    private:
        // ENVI 头文件元数据
        int m_samples = 0;      // 样本数 (宽)
        int m_lines = 0;        // 行数 (高)
        int m_bands = 0;        // 波段数
        int m_dataType = 0;     // 2=int16, 12=uint16
        int m_headerOffset = 0; // 头偏移
        std::string m_interleave; // "bil", "bip", "bsq"

        // 核心数据存储
        // 使用 uint16_t 存储原始字节，提取时根据 m_dataType 进行符号转换
        std::vector<uint16_t> m_rawData;

        // 内部解析辅助
        bool parseHeader(const std::string& hdrPath);
        bool loadBinary(const std::string& binPath);
        std::string trim(const std::string& str);
    };

} // namespace HSI_Project