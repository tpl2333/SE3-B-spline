# 相机轨迹的均匀B样条拟合 （Uniform B-spline fitting for Camera Trajectory）
本项目实现了一个基于**累积 B 样条 (Cumulative B-Spline)** 的相机轨迹拟合工具。它能够将离散的、可能带有噪声的相机姿态（位置和旋转）拟合为平滑的连续曲线。
## 功能特性
* **$SO(3)$ 旋转拟合**：在李群空间内进行旋转插值，确保旋转的平滑性与正交性。
* **$\mathbb{R}^3$ 平移拟合**：对三维空间位置进行三次样条拟合。
* **Ceres 优化**：后端使用 Google Ceres Solver 进行非线性最小二乘优化。
* **Python 绑定**：核心算法由 C++ 编写，通过 Pybind11 导出，方便 Python 调用。
* **交互式可视化**：内置基于 Open3D 的可视化工具，可直观对比原始轨迹与拟合结果。
## 项目结构
* fit_core.cpp: 核心计算引擎。包含 Ceres 代价函数定义（SO3SplineErrorTerm 和 R3SplineErrorTerm）以及 Pybind11 接口。
* fitting_curve.py: 主程序脚本。负责加载原始相机数据、初始化控制点、调用 C++ 优化模块并保存结果。
* visualize.py: 可视化工具。用于展示原始轨迹（青色）、拟合轨迹（绿色）以及控制点（橙色）。opencv_cameras.json: 输入数据示例（格式参考项目要求）。
## 环境要求
* C++ 依赖: Eigen 3、Ceres Solver、Pybind11
* Python 依赖：NumPy、SciPy、Open3D
## 编译与运行

1. **编译 C++ 核心模块**

    将 `fit_core.cpp` 编译为 Python 可调用的动态库。通常使用 CMakeLists.txt 或手动编译 需要将 `fit_core.cpp` 编译为 Python 可调用的动态库。

    ```bash
    # 创建并进入构建目录
    mkdir -p build/Release
    cd build/Release

    # 执行编译
    cmake ../..
    make
    ```
    注意：请确保 `fitting_curve.py` 中的 `from build.Release import fit_curve` 路径能正确引用生成的库文件。

2. **运行拟合程序**

    编译成功后，运行主程序进行轨迹优化。

    ```Bash
    python fitting_curve.py
    ```
    输入：读取 `opencv_cameras.json`（原始相机姿态）

    输出：生成 `fitted_spline_params.json`（优化后的控制点参数）

3. **结果可视化**

    使用 Open3D 引擎对比原始数据与拟合后的平滑曲线。

    ```Bash
    python visualize.py
    ```

    交互快捷键：
    在可视化窗口激活状态下，按下以下键位可以切换显示内容：

    A：显示/隐藏 原始观测轨迹 (Actual Path) —— 蓝色

    S：显示/隐藏 样条拟合曲线 (Spline Curve) —— 绿色

    K：显示/隐藏 控制点与骨架 (Knots/Frames) —— 橙色
