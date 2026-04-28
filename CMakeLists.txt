cmake_minimum_required(VERSION 3.10)
project(fitting_curve)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 1. 基础设置
set(CMAKE_CXX_STANDARD 14)

# 2. 寻找依赖 (vcpkg 会自动处理路径)
find_package(pybind11 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

# 3. 定义输出模块
pybind11_add_module(fit_curve src/fit_core.cpp)

# 4. 连接库
target_link_libraries(fit_curve PRIVATE Ceres::ceres Eigen3::Eigen)