#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

// 核心代价项：拟合单个观测点 R_obs
struct SO3SplineErrorTerm {

    Eigen::Quaterniond q_obs_;
    double u_;
    // 构造函数：传入观测姿态（四元数 [w,x,y,z]）和 归一化时间 u
    SO3SplineErrorTerm(const Eigen::Quaterniond& q_obs, double u)
        : q_obs_(q_obs), u_(u) {}

    template <typename T>
    bool operator()(const T* const q0_ptr, // P_{i-1}
                    const T* const q1_ptr, // P_{i}
                    const T* const q2_ptr, // P_{i+1}
                    const T* const q3_ptr, // P_{i+2}
                    T* residuals) const {

        // 1. 将原始 double 数据映射为 Ceres 可识别的模板四元数 (w, x, y, z)
        Eigen::Quaternion<T> q0(q0_ptr[0], q0_ptr[1], q0_ptr[2], q0_ptr[3]);
        Eigen::Quaternion<T> q1(q1_ptr[0], q1_ptr[1], q1_ptr[2], q1_ptr[3]);
        Eigen::Quaternion<T> q2(q2_ptr[0], q2_ptr[1], q2_ptr[2], q2_ptr[3]);
        Eigen::Quaternion<T> q3(q3_ptr[0], q3_ptr[1], q3_ptr[2], q3_ptr[3]);

        // 2. 计算累计基函数系数 (基于我们之前推导的 M_tilde 矩阵)
        // B_tilde = M_tilde^T * [1, u, u^2, u^3]^T
        T u2 = T(u_ * u_);
        T u3 = T(u2 * u_);

        // 这里就是那个 M_tilde 矩阵的各行加权
        T b_tilde1 = T(1.0/6.0) * (T(5.0) + T(3.0)*T(u_) - T(3.0)*u2 + u3);
        T b_tilde2 = T(1.0/6.0) * (T(1.0) + T(3.0)*T(u_) + T(3.0)*u2 - T(2.0)*u3); // 简化后的系数
        T b_tilde3 = T(1.0/6.0) * u3;

        // 3. 计算相对旋转增量 Omega (Log 映射)
        // Omega_i = Log(q_{i-1}^-1 * q_i)
        auto get_omega = [](const Eigen::Quaternion<T>& qa, const Eigen::Quaternion<T>& qb) {
            Eigen::Quaternion<T> q_rel = qa.inverse() * qb;
            Eigen::Matrix<T, 3, 1> omega;
            // 使用 Ceres 自带函数将四元数转为角轴 (即李代数)
            T q_array[4] = {q_rel.w(), q_rel.x(), q_rel.y(), q_rel.z()};
            ceres::QuaternionToAngleAxis(q_array, omega.data());
            return omega;
        };

        Eigen::Matrix<T, 3, 1> omega1 = get_omega(q0, q1);
        Eigen::Matrix<T, 3, 1> omega2 = get_omega(q1, q2);
        Eigen::Matrix<T, 3, 1> omega3 = get_omega(q2, q3);

        // 4. 合成预测姿态 q_pred = q0 * Exp(b1*w1) * Exp(b2*w2) * Exp(b3*w3)
        auto exp_map = [](const Eigen::Matrix<T, 3, 1>& omega_scaled) {
            T q_arr[4];
            ceres::AngleAxisToQuaternion(omega_scaled.data(), q_arr);
            return Eigen::Quaternion<T>(q_arr[0], q_arr[1], q_arr[2], q_arr[3]);
        };

        Eigen::Quaternion<T> q_pred = q0 * exp_map(b_tilde1 * omega1) 
                                         * exp_map(b_tilde2 * omega2) 
                                         * exp_map(b_tilde3 * omega3);

        // 5. 计算残差：Residual = Log(q_pred^-1 * q_obs)
        Eigen::Quaternion<T> q_obs_T = q_obs_.template cast<T>();
        Eigen::Quaternion<T> q_error = q_pred.inverse() * q_obs_T;
        
        T q_err_arr[4] = {q_error.w(), q_error.x(), q_error.y(), q_error.z()};
        ceres::QuaternionToAngleAxis(q_err_arr, residuals); // 残差占 3 维

        return true;
    }

};

struct R3SplineErrorTerm{

    Eigen::Vector3d t_obs_;
    double u_;

    // 传入观测平移 t_obs 和 归一化时间 u
    R3SplineErrorTerm(const Eigen::Vector3d& t_obs, double u)
        : t_obs_(t_obs), u_(u){}

    template<typename T>
    bool operator()(const T* const t0_ptr,
                    const T* const t1_ptr,
                    const T* const t2_ptr,
                    const T* const t3_ptr,
                    T* residuals)const{

        Eigen::Map<const Eigen::Vector3<T>> t0(t0_ptr);
        Eigen::Map<const Eigen::Vector3<T>> t1(t1_ptr);
        Eigen::Map<const Eigen::Vector3<T>> t2(t2_ptr);
        Eigen::Map<const Eigen::Vector3<T>> t3(t3_ptr);

        T u2 = T(u_ * u_);
        T u3 = T(u2 * u_);

        T b_tilde1 = T(1.0/6.0) * (T(5.0) + T(3.0)*T(u_) - T(3.0)*u2 + u3);
        T b_tilde2 = T(1.0/6.0) * (T(1.0) + T(3.0)*T(u_) + T(3.0)*u2 - T(2.0)*u3); 
        T b_tilde3 = T(1.0/6.0) * u3;

        Eigen::Vector3<T> p1 = t1 - t0;
        Eigen::Vector3<T> p2 = t2 - t1;
        Eigen::Vector3<T> p3 = t3 - t2;

        Eigen::Vector3<T> t_pred = t0 + b_tilde1 * p1 + b_tilde2 * p2 + b_tilde3 * p3;
        Eigen::Vector3<T> error = t_pred - t_obs_.cast<T>();
        Eigen::Map<Eigen::Vector3<T>> res_map(residuals);

        res_map = error;

        return true;
    }
};


void fit_single_scene(
    pybind11::array_t<double> control_rot,    // [ctrl_num, 4] 控制点四元数 (w,x,y,z) 
    pybind11::array_t<double> control_trans,  // [ctrl_num, 3] 控制点平移
    pybind11::array_t<double> obs_rot,        // [obs_num, 4] 观测四元数
    pybind11::array_t<double> obs_trans,      // [obs_num, 3] 观测平移
    pybind11::array_t<double> obs_u,          // [obs_num,] 每个观测点对应的归一化时间 u (0~1)
    pybind11::array_t<int> obs_ctrl_idx       // [obs_num,] 每个观测点对应的第一个控制点索引 (即 P_{i-1})
) {
    // 获取数据的原始指针
    pybind11::buffer_info buf_c_rot = control_rot.request();
    pybind11::buffer_info buf_c_trans = control_trans.request();
    pybind11::buffer_info buf_o_rot = obs_rot.request();
    pybind11::buffer_info buf_o_trans = obs_trans.request();
    pybind11::buffer_info buf_u = obs_u.request();
    pybind11::buffer_info buf_idx = obs_ctrl_idx.request();

    double* ptr_c_rot = static_cast<double*>(buf_c_rot.ptr);
    double* ptr_c_trans = static_cast<double*>(buf_c_trans.ptr);
    double* ptr_o_rot = static_cast<double*>(buf_o_rot.ptr);
    double* ptr_o_trans = static_cast<double*>(buf_o_trans.ptr);
    double* ptr_u = static_cast<double*>(buf_u.ptr);
    int* ptr_idx = static_cast<int*>(buf_idx.ptr);

    auto ctrl_num = buf_c_rot.shape[0]; // 控制点数量
    auto obs_num = buf_o_rot.shape[0]; // 观测点数量

    ceres::Problem problem;
    ceres::Manifold* quat_manifold = new ceres::QuaternionManifold();

    // 1. 添加参数块
    for (int i = 0; i < ctrl_num; ++i) {
        problem.AddParameterBlock(ptr_c_rot + i * 4, 4, quat_manifold);
        problem.AddParameterBlock(ptr_c_trans + i * 3, 3);
    }

    // 2. 遍历观测点，添加残差块
    for (int k = 0; k < obs_num; ++k) {
        int idx = ptr_idx[k]; // 该观测点对应的起始控制点索引
        double u = ptr_u[k];

        // 构造当前观测值的 Eigen 对象
        Eigen::Quaterniond q_obs(ptr_o_rot[k*4], ptr_o_rot[k*4+1], ptr_o_rot[k*4+2], ptr_o_rot[k*4+3]);
        Eigen::Vector3d t_obs(ptr_o_trans[k*3], ptr_o_trans[k*3+1], ptr_o_trans[k*3+2]);

        // 获取 4 个控制点的内存地址
        double* r0 = ptr_c_rot + idx * 4;
        double* r1 = ptr_c_rot + (idx + 1) * 4;
        double* r2 = ptr_c_rot + (idx + 2) * 4;
        double* r3 = ptr_c_rot + (idx + 3) * 4;

        double* t0 = ptr_c_trans + idx * 3;
        double* t1 = ptr_c_trans + (idx + 1) * 3;
        double* t2 = ptr_c_trans + (idx + 2) * 3;
        double* t3 = ptr_c_trans + (idx + 3) * 3;

        // 添加 SO3 残差
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SO3SplineErrorTerm, 3, 4, 4, 4, 4>(
                new SO3SplineErrorTerm(q_obs, u)),
            nullptr, r0, r1, r2, r3);

        // 添加 R3 残差
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<R3SplineErrorTerm, 3, 3, 3, 3, 3>(
                new R3SplineErrorTerm(t_obs, u)),
            nullptr, t0, t1, t2, t3);
    }

    // 3. 配置求解器并优化
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = false; // 关掉输出，否则终端会刷屏

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

// Pybind11 模块导出
PYBIND11_MODULE(fit_curve, m) {
    m.def("fit_single_scene", &fit_single_scene, "Fit B-Spline for a single scene");
}

