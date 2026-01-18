#include "SpeLoader.h"
#include <sstream>
#include <algorithm>
#include <filesystem>

namespace HSI_Project {

    SpeLoader::SpeLoader() {}
    SpeLoader::~SpeLoader() {}

    std::string SpeLoader::trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (std::string::npos == first) return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }

    bool SpeLoader::load(const std::string& filePath) {
        std::string hdrPath = filePath;
        if (hdrPath.find(".hdr") == std::string::npos) {
            hdrPath += ".hdr";
        }

        if (!parseHeader(hdrPath)) return false;

        // 查找二进制数据文件
        std::string binPath = hdrPath.substr(0, hdrPath.find(".hdr"));
        if (!std::filesystem::exists(binPath)) {
            if (std::filesystem::exists(binPath + ".spe")) binPath += ".spe";
            else if (std::filesystem::exists(binPath + ".raw")) binPath += ".raw";
            else {
                std::cerr << "[SpeLoader] Binary file not found for: " << binPath << std::endl;
                return false;
            }
        }

        return loadBinary(binPath);
    }

    bool SpeLoader::parseHeader(const std::string& hdrPath) {
        std::ifstream file(hdrPath);
        if (!file.is_open()) return false;

        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            size_t eqPos = line.find('=');
            if (eqPos != std::string::npos) {
                std::string key = trim(line.substr(0, eqPos));
                std::string value = trim(line.substr(eqPos + 1));
                std::transform(key.begin(), key.end(), key.begin(), ::tolower);

                if (key == "samples") m_samples = std::stoi(value);
                else if (key == "lines") m_lines = std::stoi(value);
                else if (key == "bands") m_bands = std::stoi(value);
                else if (key == "header offset") m_headerOffset = std::stoi(value);
                else if (key == "data type") m_dataType = std::stoi(value);
                else if (key == "interleave") {
                    m_interleave = value;
                    std::transform(m_interleave.begin(), m_interleave.end(), m_interleave.begin(), ::tolower);
                }
            }
        }

        // 核心微调：明确支持 ENVI Data Type 2 (Signed 16-bit integer)
        if (m_dataType != 2 && m_dataType != 12) {
            std::cerr << "[SpeLoader] Critical: This loader is tuned for Data Type 2/12. Found: " << m_dataType << std::endl;
            return false;
        }

        return (m_samples > 0 && m_lines > 0 && m_bands > 0);
    }

    bool SpeLoader::loadBinary(const std::string& binPath) {
        std::ifstream file(binPath, std::ios::binary);
        if (!file.is_open()) return false;

        file.seekg(m_headerOffset, std::ios::beg);

        size_t totalElements = static_cast<size_t>(m_samples) * m_lines * m_bands;
        m_rawData.resize(totalElements);

        // 完美兼容 Data Type 2：按 2 字节（16位）块读取
        // 即使是 signed int16，在读取原始位数据时也可以先存入 uint16 容器
        file.read(reinterpret_cast<char*>(m_rawData.data()), totalElements * sizeof(uint16_t));

        if (!file) {
            std::cerr << "[SpeLoader] Error: Incomplete binary read. Expected " << totalElements << " elements." << std::endl;
            return false;
        }
        file.close();
        return true;
    }

    std::vector<float> SpeLoader::getSpectrum(int x, int y) const {
        std::vector<float> spectrum(m_bands, 0.0f);
        if (x < 0 || x >= m_samples || y < 0 || y >= m_lines) return spectrum;

        long long baseIndex = 0;
        for (int b = 0; b < m_bands; ++b) {
            // 针对 BIL (Band Interleaved by Line) 格式的索引优化
            if (m_interleave == "bil") {
                baseIndex = (static_cast<long long>(y) * m_bands * m_samples) + 
                            (static_cast<long long>(b) * m_samples) + x;
            } 
            else if (m_interleave == "bip") {
                baseIndex = (static_cast<long long>(y) * m_samples + x) * m_bands + b;
            } 
            else { // BSQ
                baseIndex = (static_cast<long long>(b) * m_lines * m_samples) + 
                            (static_cast<long long>(y) * m_samples) + x;
            }

            if (baseIndex < m_rawData.size()) {
                // 如果是 Data Type 2 (Signed)，此处会自动处理符号转换
                // 如果原始值是 int16_t，转为 float 会保留其数值逻辑
                spectrum[b] = static_cast<float>(static_cast<int16_t>(m_rawData[baseIndex]));
            }
        }
        return spectrum;
    }

} // namespace HSI_Project