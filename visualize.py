import json
import numpy as np
import open3d as o3d

# === B-样条基函数 (累积形式，用于平滑插值) ===
def get_cumulative_basis(u):
    u2 = u * u
    u3 = u2 * u
    b1 = (5.0 + 3.0*u - 3.0*u2 + u3) / 6.0
    b2 = (1.0 + 3.0*u + 3.0*u2 - 2.0*u3) / 6.0
    b3 = u3 / 6.0
    return b1, b2, b3

def interpolate_translation(p0, p1, p2, p3, u):
    """给定4个控制点和归一化时间u，计算插值位置"""
    b1, b2, b3 = get_cumulative_basis(u)
    return p0 + b1*(p1 - p0) + b2*(p2 - p1) + b3*(p3 - p2)

def quaternion_to_rotation_matrix(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ])

class SplineVisualizer:
    def __init__(self, fitted_json, original_json):
        self.geometries = {}
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        
        # 状态位
        self.show_actual = True
        self.show_spline = True
        self.show_knots = True

        self.load_data(fitted_json, original_json)
        self.setup_ui()

    def load_data(self, fitted_json, original_json):
        # 1. 世界坐标轴
        self.geometries['world'] = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        # 2. 拟合后的数据
        with open(fitted_json, 'r') as f:
            f_data = json.load(f)
        cps = f_data['control_points']
        cp_pts = np.array([cp['t'] for cp in cps])
        interval = f_data['interpolation_interval']

        # 控制点小坐标架 + 骨架 (橙色)
        self.geometries['knots_frames'] = []
        for cp in cps:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            T = np.eye(4)
            T[:3, :3] = quaternion_to_rotation_matrix(cp['q_wxyz'])
            T[:3, 3] = cp['t']
            frame.transform(T)
            self.geometries['knots_frames'].append(frame)

        lines = [[i, i+1] for i in range(len(cp_pts)-1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(cp_pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[1, 0.5, 0] for _ in range(len(lines))])
        self.geometries['knots_line'] = ls

        # 3. 生成密集的样条曲线 (绿色)
        spline_pts = []
        # 遍历每一段 (从 P0 开始到 P_{K-1})
        # 数组里索引是 [P-1, P0, P1, ..., PK, PK+1]
        # 对应数组下标 [0, 1, 2, ..., len-2, len-1]
        for i in range(1, len(cp_pts) - 2):
            p0, p1, p2, p3 = cp_pts[i-1], cp_pts[i], cp_pts[i+1], cp_pts[i+2]
            for u in np.linspace(0, 1, 10): # 每段插值10个点
                spline_pts.append(interpolate_translation(p0, p1, p2, p3, u))
        
        spline_pts = np.array(spline_pts)
        s_lines = [[i, i+1] for i in range(len(spline_pts)-1)]
        sls = o3d.geometry.LineSet()
        sls.points = o3d.utility.Vector3dVector(spline_pts)
        sls.lines = o3d.utility.Vector2iVector(s_lines)
        sls.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(s_lines))]) # 鲜绿色
        self.geometries['spline_line'] = sls

        # 4. 原始观测轨迹 (青色)
        with open(original_json, 'r') as f:
            o_data = json.load(f)
        obs_pts = []
        for frame in o_data['frames']:
            w2c = np.array(frame['w2c'])
            c2w = np.linalg.inv(w2c)
            obs_pts.append(c2w[:3, 3])
        
        obs_pts = np.array(obs_pts)
        o_lines = [[i, i+1] for i in range(len(obs_pts)-1)]
        ols = o3d.geometry.LineSet()
        ols.points = o3d.utility.Vector3dVector(obs_pts)
        ols.lines = o3d.utility.Vector2iVector(o_lines)
        ols.colors = o3d.utility.Vector3dVector([[0, 1, 1] for _ in range(len(o_lines))]) # 青色
        self.geometries['actual_line'] = ols

    def toggle_actual(self, vis):
        self.show_actual = not self.show_actual
        if self.show_actual: vis.add_geometry(self.geometries['actual_line'], reset_bounding_box=False)
        else: vis.remove_geometry(self.geometries['actual_line'], reset_bounding_box=False)
        return False

    def toggle_spline(self, vis):
        self.show_spline = not self.show_spline
        if self.show_spline: vis.add_geometry(self.geometries['spline_line'], reset_bounding_box=False)
        else: vis.remove_geometry(self.geometries['spline_line'], reset_bounding_box=False)
        return False

    def toggle_knots(self, vis):
        self.show_knots = not self.show_knots
        if self.show_knots:
            vis.add_geometry(self.geometries['knots_line'], reset_bounding_box=False)
            for f in self.geometries['knots_frames']: vis.add_geometry(f, reset_bounding_box=False)
        else:
            vis.remove_geometry(self.geometries['knots_line'], reset_bounding_box=False)
            for f in self.geometries['knots_frames']: vis.remove_geometry(f, reset_bounding_box=False)
        return False

    def setup_ui(self):
        self.vis.create_window(window_name="Spline Comparison (A: Actual, S: Spline, K: Knots)", width=1280, height=720)
        
        # 初始添加全部
        self.vis.add_geometry(self.geometries['world'])
        self.vis.add_geometry(self.geometries['actual_line'])
        self.vis.add_geometry(self.geometries['spline_line'])
        self.vis.add_geometry(self.geometries['knots_line'])
        for f in self.geometries['knots_frames']: self.vis.add_geometry(f)

        # 注册快捷键 (A: 65, S: 83, K: 75)
        self.vis.register_key_callback(65, self.toggle_actual)
        self.vis.register_key_callback(83, self.toggle_spline)
        self.vis.register_key_callback(75, self.toggle_knots)

        print("Controls: [A] Toggle Actual Path, [S] Toggle Spline Curve, [K] Toggle Knots")
        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    FITTED = "fitted_spline_params.json"
    ORIGINAL = "opencv_cameras.json"
    SplineVisualizer(FITTED, ORIGINAL)