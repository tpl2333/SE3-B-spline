import json
import numpy as np

from build.Release import fit_curve  
from scipy.spatial.transform import Rotation as R

def load_camera_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    obs_rot = []
    obs_trans = []
    
    for frame in frames:
        # 提取 w2c 矩阵 (4x4)
        w2c = np.array(frame['w2c'])
        
        # 为了拟合轨迹，通常使用 c2w (Camera-to-World)
        # c2w = inv(w2c)
        c2w = np.linalg.inv(w2c)
        
        rot_matrix = c2w[:3, :3]
        trans_vector = c2w[:3, 3]
        
        # 转换为四元数 [w, x, y, z] -> Eigen 默认顺序
        # scipy 默认是 [x, y, z, w]，需要重排
        quat_xyzw = R.from_matrix(rot_matrix).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        obs_rot.append(quat_wxyz)
        obs_trans.append(trans_vector)
        
    return np.array(obs_rot), np.array(obs_trans), data['scene_name']

def solve_spline(obs_rot, obs_trans, interval=8):
    obs_num = len(obs_rot)
    
    # 1. 准备观测点的 u 和起始控制点索引 idx
    # idx = k // interval, u = (k % interval) / interval
    obs_u = (np.arange(obs_num) % interval) / float(interval)
    obs_ctrl_idx = (np.arange(obs_num) // interval).astype(np.int32)
    
    # 2. 初始化控制点
    # 控制点数量 ctrl_num = (obs_num)//interval + 1 + 3 (前面补一个，后面补两个)
    max_idx = obs_ctrl_idx[-1]
    ctrl_num = max_idx + 4
    
    control_rot = np.zeros((ctrl_num, 4))
    control_trans = np.zeros((ctrl_num, 3))
    
    # 简单初始化：每隔 interval 帧取一个观测值作为控制点
    for i in range(max_idx+1):
        sample_idx = i * interval
        control_rot[i+1] = obs_rot[sample_idx]
        control_trans[i+1] = obs_trans[sample_idx]
    
    # 边界填充
    control_rot[0] = rot_linear_padding(control_rot[2],control_rot[1])
    control_rot[-2] = rot_linear_padding(control_rot[-4], control_rot[-3])
    control_rot[-1] = rot_linear_padding(control_rot[-3], control_rot[-2])

    control_trans[0] = control_trans[1] + control_trans[1] - control_trans[2]
    control_trans[-2] = control_trans[-3] + control_trans[-3] - control_trans[-4]
    control_trans[-1] = control_trans[-2] + control_trans[-2] - control_trans[-3]

    # 3. 调用 C++ 核心进行优化
    # 注意：fit_single_scene 会直接修改 control_rot 和 control_trans 的内容
    print(f"Starting optimization for {obs_num} frames with {ctrl_num} control points...")
    fit_curve.fit_single_scene(
        control_rot, 
        control_trans, 
        obs_rot, 
        obs_trans, 
        obs_u, 
        obs_ctrl_idx
    )
    print("Optimization finished.")
    
    return control_rot, control_trans

def rot_linear_padding(quat_wxyz1, quat_wxyz2):

    quat1 = R.from_quat(np.array([quat_wxyz1[1], quat_wxyz1[2], quat_wxyz1[3], quat_wxyz1[0]]))
    quat2 = R.from_quat(np.array([quat_wxyz2[1], quat_wxyz2[2], quat_wxyz2[3], quat_wxyz2[0]]))

    padding_quat = quat2 * (quat1.inv() * quat2)
    quat_xyzw = padding_quat.as_quat()
    
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
def save_params(scene_name, ctrl_rot, ctrl_trans, interval, output_path):
    # 将结果保存为 JSON
    res = {
        "scene_name": scene_name,
        "interpolation_interval": interval,
        "control_points": []
    }
    
    for i in range(len(ctrl_rot)):
        res["control_points"].append({
            "id": i,
            "q_wxyz": ctrl_rot[i].tolist(),
            "t": ctrl_trans[i].tolist()
        })
        
    with open(output_path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    INPUT_JSON = "./opencv_cameras.json"
    OUTPUT_JSON = "./fitted_spline_params.json"
    INTERVAL = 8
    
    # 1. 加载数据
    obs_rot, obs_trans, scene_name = load_camera_data(INPUT_JSON)
    
    # 2. 拟合轨迹
    ctrl_rot, ctrl_trans = solve_spline(obs_rot, obs_trans, interval=INTERVAL)
    
    # 3. 导出参数
    save_params(scene_name, ctrl_rot, ctrl_trans, INTERVAL, OUTPUT_JSON)