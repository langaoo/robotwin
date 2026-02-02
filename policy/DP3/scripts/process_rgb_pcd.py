import os
import numpy as np
import h5py
import cv2
import argparse
import shutil
import json

def save_ply(points, save_path):
    """保存点云为PLY格式（二进制优化版，解决写入过慢问题）"""
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    if points.size == 0:
        # 允许保存空点云，保持文件索引对齐
        pass

    # 解析属性
    has_color = points.shape[1] >= 6
    has_uv = points.shape[1] >= 8
    
    # 准备数据结构
    # XYZ (float32)
    xyz = points[:, :3].astype(np.float32)
    
    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    
    if has_color:
        # RGB (uchar) - 假设输入是 0-1 float
        rgb = (points[:, 3:6] * 255).astype(np.uint8)
        dtype_list.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        
    if has_uv:
        # UV (float32)
        uv = points[:, 6:8].astype(np.float32)
        dtype_list.extend([('u', 'f4'), ('v', 'f4')])

    # 构建结构化数组
    vertex_data = np.empty(len(points), dtype=dtype_list)
    vertex_data['x'] = xyz[:, 0]
    vertex_data['y'] = xyz[:, 1]
    vertex_data['z'] = xyz[:, 2]
    
    if has_color:
        vertex_data['red'] = rgb[:, 0]
        vertex_data['green'] = rgb[:, 1]
        vertex_data['blue'] = rgb[:, 2]
        
    if has_uv:
        vertex_data['u'] = uv[:, 0]
        vertex_data['v'] = uv[:, 1]

    with open(save_path, 'wb') as f:
        # Header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(points)}\n".encode('ascii'))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if has_color:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        if has_uv:
            f.write(b"property float u\n")
            f.write(b"property float v\n")
        f.write(b"end_header\n")
        
        # Data
        vertex_data.tofile(f)


def save_camera_info_json(save_path, intrinsic, cam2world_gl, image_hw=None):
    """保存每帧相机信息到json"""
    payload = {
        "intrinsic_cv": np.asarray(intrinsic).tolist(),
        "cam2world_gl": np.asarray(cam2world_gl).tolist(),
    }
    if image_hw is not None:
        payload["image_hw"] = [int(image_hw[0]), int(image_hw[1])]
    with open(save_path, "w") as f:
        json.dump(payload, f)

def depth_to_pointcloud(depth, rgb_img, intrinsic, cam2world_gl, depth_max=2.0):
    """从深度图和RGB图生成密集点云（世界坐标系）
    
    使用OpenGL坐标系约定（与Sapien一致）:
    - 相机坐标系：X右，Y上，Z向后（物体在前方时Z<0）
    - depth = -position[..., 2] * 1000（Sapien保存方式）
    
    关键优化：
    - 过滤远处背景点（depth_max阈值）
    - 过滤深度不连续的边缘点（物体边界噪声）
    
    参数:
        depth: 深度图 (H, W)，单位为毫米
        rgb_img: RGB格式图像 (H, W, 3)，范围0-255（注意：必须是RGB格式，不是BGR）
        intrinsic: 相机内参矩阵 (3, 3)
        cam2world_gl: 相机到世界的变换矩阵 (4, 4)，OpenGL坐标系
        depth_max: 最大深度阈值（米），用于过滤远处背景点（默认2.0米）
    """
    h, w = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # OpenGL相机坐标系：Z向后，物体在前方时Z<0
    # depth = -position_z * 1000，所以 position_z = -depth / 1000
    z_opengl = -depth / 1000.0  # 注意负号！
    # 🔧 关键修复：X坐标需要取反以匹配Sapien Position buffer
    # Sapien存储的depth图已经镜像了X方向（图像u增大对应世界X减小）
    x = -(u - cx) * z_opengl / fx  # 注意负号！
    y = (v - cy) * z_opengl / fy
    
    # 齐次坐标
    points_cam_homo = np.stack([x, y, z_opengl, np.ones_like(x)], axis=-1).reshape(-1, 4)
    
    # 关键优化1：过滤无效深度点和远处背景点
    depth_in_meters = depth / 1000.0
    valid_mask_depth = (depth > 0) & (depth_in_meters < depth_max)
    
    # 关键优化2：过滤深度不连续的边缘点（物体边界噪声）
    # 使用Sobel算子计算梯度的模
    sobel_x = cv2.Sobel(depth_in_meters, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_in_meters, cv2.CV_64F, 0, 1, ksize=3)
    depth_diff = np.abs(sobel_x) + np.abs(sobel_y)
    # 过滤深度梯度过大的点（边缘噪声）
    edge_mask = depth_diff < 0.05  # 5cm梯度阈值
    
    # 合并mask
    valid_mask = (valid_mask_depth & edge_mask).flatten()
    points_cam_homo = points_cam_homo[valid_mask]

    # 对应点的像素坐标 (u,v)
    u_orig, v_orig = np.meshgrid(np.arange(w), np.arange(h))
    uv = np.stack([u_orig, v_orig], axis=-1).reshape(-1, 2)[valid_mask].astype(np.float32)
    
    # 转换到世界坐标系
    points_world_homo = (cam2world_gl @ points_cam_homo.T).T
    points_world = points_world_homo[:, :3]
    
    # 用户最终决策: 完全不裁剪！让模型自己学习场景的完整信息
    # 坐标系修复后，不应该需要bbox裁剪来弥补偏移
    # 裁剪会导致：
    # 1. 丢失25%的训练数据 (4730 → 3533)
    # 2. RGBD只有17%点在物体上 vs Sapien 100%
    # 3. 破坏点云的自然分布，影响模型泛化能力
    
    # 添加颜色信息
    rgb_reshaped = rgb_img.reshape(-1, 3)[valid_mask]
    
    # 对应的UV坐标
    u_orig, v_orig = np.meshgrid(np.arange(w), np.arange(h))
    uv = np.stack([u_orig, v_orig], axis=-1).reshape(-1, 2)[valid_mask].astype(np.float32)
    
    pointcloud_with_color = np.concatenate([points_world, rgb_reshaped / 255.0, uv], axis=1)
    
    return pointcloud_with_color

def load_hdf5(dataset_path):
    """读取HDF5文件中的点云和RGB数据"""
    if not os.path.isfile(dataset_path):
        print(f"文件不存在: {dataset_path}")
        return None, None, None
    
    with h5py.File(dataset_path, "r") as root:
        # 读取点云数据
        pointcloud = root["/pointcloud"][()] if "/pointcloud" in root else None
        
        # 读取所有摄像头的RGB数据
        rgb_dict = {}
        depth_dict = {}
        intrinsic_dict = {}
        cam2world_gl_dict = {}  # 使用OpenGL坐标系的变换矩阵
        
        if "/observation" in root:
            for cam_name in root["/observation"].keys():
                cam_group = root[f"/observation/{cam_name}"]
                if "rgb" in cam_group:
                    rgb_bytes = cam_group["rgb"][()]
                    rgb_dict[cam_name] = rgb_bytes
                if "depth" in cam_group:
                    depth_dict[cam_name] = cam_group["depth"][()]
                if "intrinsic_cv" in cam_group:
                    intrinsic_dict[cam_name] = cam_group["intrinsic_cv"][()]
                if "cam2world_gl" in cam_group:
                    cam2world_gl_dict[cam_name] = cam_group["cam2world_gl"][()]
    
    return pointcloud, rgb_dict, (depth_dict, intrinsic_dict, cam2world_gl_dict)

def main():
    parser = argparse.ArgumentParser(description="提取点云(PLY)和RGB(PNG)并按结构保存")
    parser.add_argument("task_name", type=str, help="任务名称（如beat_block_hammer）")
    parser.add_argument("task_config", type=str, help="任务配置（如demo_randomized）")
    parser.add_argument("expert_data_num", type=int, help="需要处理的episode数量")
    parser.add_argument("--output_root", type=str, default="/home/gl/RoboTwin/policy/DP2DP3/features_model",
                        help="输出根目录（默认：/home/gl/RoboTwin/policy/DP2DP3/features_model）")
    parser.add_argument("--use_dense", action="store_true",
                        help="使用depth+RGB生成密集点云")
    parser.add_argument("--dense_camera", type=str, default="all",
                        help="使用哪个相机生成密集点云(front_camera/head_camera/left_camera/right_camera/all)")
    args = parser.parse_args()

    # 输入数据路径（HDF5文件所在目录）
    input_data_dir = os.path.join("../../data", args.task_name, args.task_config, "data")
    # 输出根目录
    base_root = args.output_root

    # 任务子目录名
    task_subdir = f"{args.task_name}-{args.task_config}-{args.expert_data_num}"

    # 处理每个episode
    for ep in range(args.expert_data_num):
        hdf5_path = os.path.join(input_data_dir, f"episode{ep}.hdf5")
        print(f"处理 episode {ep}/{args.expert_data_num}：{hdf5_path}")

        # 读取数据
        pointcloud_all, rgb_dict, depth_data = load_hdf5(hdf5_path)
        
        # 检查文件是否存在或读取失败
        if depth_data is None:
            print(f"警告：episode {ep} 文件不存在或读取失败，跳过")
            continue
            
        depth_dict, intrinsic_dict, cam2world_gl_dict = depth_data
        
        if pointcloud_all is None and not rgb_dict:
            print(f"警告：episode {ep} 无有效数据，跳过")
            continue

        # 处理点云（每个步骤保存为一个PLY）
        if args.use_dense and depth_dict and rgb_dict:
            # 确定要处理的相机列表
            if args.dense_camera == "all":
                cameras_to_process = list(depth_dict.keys())
            else:
                cameras_to_process = [args.dense_camera]
            
            # 为每个相机生成独立的点云文件
            for cam_name in cameras_to_process:
                if cam_name not in depth_dict:
                    print(f"警告：相机{cam_name}不存在，跳过")
                    continue
                
                # 每个相机一个独立的目录 (保存到 features_model/pc_dataset/PC 和 rgb_dataset/RGB)
                pcd_root = os.path.join(base_root, "pc_dataset", "PC", f"{task_subdir}_{cam_name}")
                rgb_root = os.path.join(base_root, "rgb_dataset", "RGB", f"{task_subdir}_{cam_name}")
                os.makedirs(pcd_root, exist_ok=True)
                os.makedirs(rgb_root, exist_ok=True)
                
                # 创建当前episode的文件夹
                ep_pcd_dir = os.path.join(pcd_root, f"episode_{ep}")
                os.makedirs(ep_pcd_dir, exist_ok=True)
                
                depth_data_cam = depth_dict[cam_name]
                rgb_bytes_list = rgb_dict[cam_name]
                intrinsic_all = intrinsic_dict[cam_name]
                cam2world_gl_all = cam2world_gl_dict[cam_name]
                
                print(f"  正在处理相机 {cam_name} (共 {len(depth_data_cam)} 帧)...")
                
                # 准备 RGB 输出目录
                cam_rgb_dir = os.path.join(rgb_root, f"episode_{ep}")
                os.makedirs(cam_rgb_dir, exist_ok=True)

                # 确保帧数对齐
                n_frames = min(len(depth_data_cam), len(rgb_bytes_list))
                
                for step in range(n_frames):
                    if step % 50 == 0:
                        print(f"    处理进度: {step}/{n_frames}")

                    # 1. 解码图像并转换为RGB
                    # 注意: HDF5中JPEG数据实际上以BGR通道编码，使用PIL解码后反转通道更可靠
                    from PIL import Image
                    import io
                    img_pil = Image.open(io.BytesIO(rgb_bytes_list[step]))
                    img_array = np.array(img_pil)
                    # 通道反转: BGR -> RGB
                    img_rgb = img_array[:, :, ::-1]
                    
                    # 2. 生成密集点云 (使用OpenGL坐标系)
                    dense_pc = depth_to_pointcloud(
                        depth_data_cam[step], 
                        img_rgb,  # 传入RGB格式
                        intrinsic_all[step], 
                        cam2world_gl_all[step]
                    )
                    
                    ply_path = os.path.join(ep_pcd_dir, f"step_{step:04d}.ply")
                    save_ply(dense_pc, ply_path)

                    # 保存相机信息（保存GL矩阵用于调试）
                    cam_info_path = os.path.join(ep_pcd_dir, f"step_{step:04d}.ply.camera.json")
                    save_camera_info_json(
                        cam_info_path,
                        intrinsic_all[step],
                        cam2world_gl_all[step],
                        image_hw=depth_data_cam[step].shape,
                    )
                    
                    # 3. 保存 RGB 图像 (修正颜色通道)
                    # 🔧 修复: HDF5中是BGR，已转换为RGB，cv2.imwrite保存BGR，所以需要再转回BGR
                    # 但更简单的是用PIL保存RGB格式
                    from PIL import Image
                    png_path = os.path.join(cam_rgb_dir, f"step_{step:04d}.png")
                    Image.fromarray(img_rgb).save(png_path)  # 直接保存RGB格式
                    
        elif pointcloud_all is not None:
            # 使用原始稀疏点云
            pcd_root = os.path.join(base_root, "pc_dataset", "PC", task_subdir)
            rgb_root = os.path.join(base_root, "rgb_dataset", "RGB", task_subdir)
            os.makedirs(pcd_root, exist_ok=True)
            os.makedirs(rgb_root, exist_ok=True)
            
            ep_pcd_dir = os.path.join(pcd_root, f"episode_{ep}")
            os.makedirs(ep_pcd_dir, exist_ok=True)
            
            for step in range(len(pointcloud_all)):
                ply_path = os.path.join(ep_pcd_dir, f"step_{step:04d}.ply")
                save_ply(pointcloud_all[step], ply_path)
            
            # 保存RGB图像（修正颜色通道）
            if rgb_dict:
                from PIL import Image
                import io
                for cam_name, rgb_bytes_list in rgb_dict.items():
                    cam_rgb_dir = os.path.join(rgb_root, f"episode_{ep}", cam_name)
                    os.makedirs(cam_rgb_dir, exist_ok=True)
                    for step in range(len(rgb_bytes_list)):
                        # 🔧 修复: 使用PIL解码JPEG，然后通道反转
                        # JPEG编码时存的是BGR顺序，PIL解码为RGB后需要反转通道
                        img_pil = Image.open(io.BytesIO(rgb_bytes_list[step]))
                        img_array = np.array(img_pil)
                        img_rgb = img_array[:, :, ::-1]  # 通道反转: BGR -> RGB
                        png_path = os.path.join(cam_rgb_dir, f"step_{step:04d}.png")
                        Image.fromarray(img_rgb).save(png_path)

    print(f"所有数据处理完成，保存至：{base_root}")

if __name__ == "__main__":
    main()