import open3d as o3d
import numpy as np
import argparse
from pathlib import Path

def visualize_point_cloud(points, window_name="3D Point Cloud Visualization", point_size=2):
    """
    3D可视化点云数据
    
    参数:
        points: 点云数据，格式为numpy数组，形状为 (N, 3) 或 (N, 6)
                - (N, 3): 仅包含xyz坐标
                - (N, 6): 包含xyz坐标 + rgb颜色（范围0-1或0-255）
        window_name: 可视化窗口名称
        point_size: 点的显示大小
    """
    # 检查输入格式
    if len(points.shape) != 2 or points.shape[1] not in (3, 6):
        raise ValueError("点云数据必须是形状为 (N, 3) 或 (N, 6) 的二维数组")

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 设置坐标
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # 如果包含颜色信息，设置颜色
    if points.shape[1] == 6:
        colors = points[:, 3:6]
        # 处理颜色范围（如果是0-255则转换为0-1）
        if np.max(colors) > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    # 添加点云到窗口
    vis.add_geometry(pcd)
    
    # 设置点大小
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = [0.0, 0.0, 0.0]  # 黑色背景
    
    # 启动可视化（按ESC退出）
    vis.run()
    vis.destroy_window()

def load_point_cloud(file_path):
    """从文件加载点云（支持PLY、PCD等格式）"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    pcd = o3d.io.read_point_cloud(str(file_path))
    if not pcd.has_points():
        raise ValueError(f"无法读取点云数据: {file_path}")
    
    # 转换为numpy数组（包含颜色时为N×6，否则为N×3）
    points = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        points = np.hstack([points, colors])
    return points

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="3D点云可视化工具")
    parser.add_argument("--file", type=str, help="点云文件路径（支持PLY、PCD等格式）")
    parser.add_argument("--size", type=int, default=2, help="点的显示大小")
    args = parser.parse_args()

    try:
        if args.file:
            # 从文件加载并可视化
            print(f"加载点云文件: {args.file}")
            points = load_point_cloud(args.file)
            visualize_point_cloud(points, point_size=args.size)
        else:
            # 生成示例点云（螺旋线形状）
            print("未指定文件，显示示例点云")
            t = np.linspace(0, 10 * np.pi, 1000)
            x = t * np.cos(t) / 10
            y = t * np.sin(t) / 10
            z = t / 10
            # 生成颜色（随z值变化）
            colors = np.zeros((len(t), 3))
            colors[:, 0] = (np.sin(z) + 1) / 2  # 红色通道
            colors[:, 1] = (np.cos(z) + 1) / 2  # 绿色通道
            colors[:, 2] = 0.5  # 蓝色通道固定
            # 组合坐标和颜色
            example_points = np.column_stack([x, y, z, colors])
            visualize_point_cloud(example_points, point_size=5)
    except Exception as e:
        print(f"错误: {e}")