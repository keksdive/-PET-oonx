cmake_minimum_required(VERSION 3.10)
project(HSI_PET_Sorter)

set(CMAKE_CXX_STANDARD 17)

# --- 1. OpenCV 配置 ---
# [请修改] 设置你的 OpenCV 安装路径（取消注释并修改）
# set(OpenCV_DIR "D:\opencv\opencv\build\bin")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- 2. ONNX Runtime 配置 ---
# [请修改] 设置你的 ONNX Runtime 安装目录（包含 include/lib）
# set(ONNX_ROOT "I:\ONNXRUNTIME\onnxruntime-win-x64-gpu-1.23.2")

# include_directories(${ONNX_ROOT}/include)
# link_directories(${ONNX_ROOT}/lib)

# --- 3. 源文件收集 ---
file(GLOB SOURCES "src/*.cpp")

add_executable(HSI_App ${SOURCES})

# --- 4. 链接库 ---
target_link_libraries(HSI_App ${OpenCV_LIBS})
# target_link_libraries(HSI_App onnxruntime)

# --- 5. 自动复制 DLL ---
# add_custom_command(TARGET HSI_App POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#     "${ONNX_ROOT}/lib/onnxruntime.dll"
#     "$<TARGET_FILE_DIR:HSI_App>"
# )
