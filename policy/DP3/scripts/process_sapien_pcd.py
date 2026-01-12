"""
从原生HDF5数据中提取Sapien点云并保存为PLY格式
同时提取对应的RGB图像，自动截断空点云帧

与process_rgb_pcd.py对应，但直接提取存储的点云数据

使用示例:
    python scripts/process_sapien_pcd.py dump_bin_bigbin demo_randomized 20 \
        --output_root ./rgbpc_dataset --camera head_camera
"""

import os
import sys
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image


def save_ply(pointcloud, output_path):
    """保存点云为PLY格式
    
    Args:
        pointcloud: (N, 6) 数组，格式为 [x, y, z, r, g, b]
        output_path: 输出PLY文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    xyz = pointcloud[:, :3]
    rgb = (pointcloud[:, 3:6] * 255).astype(np.uint8)  # 假设RGB在0-1范围
    
    # PLY header
    header = f"""ply
format ascii 1.0
element vertex {len(xyz)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # 写入PLY文件
    with open(output_path, 'w') as f:
        f.write(header)
        for i in range(len(xyz)):
            f.write(f"{xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]} {rgb[i, 0]} {rgb[i, 1]} {rgb[i, 2]}\n")
    
    # 不打印每个文件，太多了
    # print(f"Saved: {output_path} ({len(xyz)} points)")


def save_rgb(rgb_data, output_path):
    """保存RGB图像为PNG格式
    
    Args:
        rgb_data: RGB数据，可能是numpy数组(H,W,3)或JPEG bytes
        output_path: 输出PNG文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 检查数据类型
    if isinstance(rgb_data, bytes):
        # 如果是JPEG bytes，直接解码
        import io
        img = Image.open(io.BytesIO(rgb_data))
        img.save(output_path)
    elif isinstance(rgb_data, np.ndarray):
        # 如果是numpy数组
        if rgb_data.dtype != np.uint8:
            rgb_data = rgb_data.astype(np.uint8)
        img = Image.fromarray(rgb_data, mode='RGB')
        img.save(output_path)
    else:
        # 尝试转换为数组
        try:
            rgb_array = np.array(rgb_data, dtype=np.uint8)
            img = Image.fromarray(rgb_array, mode='RGB')
            img.save(output_path)
        except:
            # 如果是JPEG二进制数据的类似bytes对象
            import io
            img = Image.open(io.BytesIO(bytes(rgb_data)))
            img.save(output_path)


def process_sapien_pointcloud(task_name, tag, num_episodes, output_root="./rgbpc_dataset", camera_name="head_camera"):
    """从HDF5提取Sapien原生点云并保存为PLY，同时提取RGB图像
    
    Args:
        task_name: 任务名称 (e.g., dump_bin_bigbin)
        tag: 数据标签 (e.g., demo_randomized)
        num_episodes: 要处理的episode数量
        output_root: 输出根目录
        camera_name: 相机名称 (e.g., head_camera)
    """
    # 构建路径
    data_dir = Path(f"../../data/{task_name}/{tag}/data")
    # 使用相机名称作为后缀，与RGBD区分
    output_pc_dir = Path(output_root) / "PC" / f"{task_name}-{tag}-{num_episodes}_sapien_{camera_name}"
    output_rgb_dir = Path(output_root) / "RGB" / f"{task_name}-{tag}-{num_episodes}_sapien_{camera_name}"
    
    print(f"📂 数据目录: {data_dir}")
    print(f"📂 点云输出: {output_pc_dir}")
    print(f"📂 RGB输出: {output_rgb_dir}")
    
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 创建输出目录
    output_pc_dir.mkdir(parents=True, exist_ok=True)
    output_rgb_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_frames = 0
    empty_count = 0
    truncated_episodes = []
    
    # 处理每个episode
    for episode_idx in tqdm(range(num_episodes), desc="Processing episodes"):
        hdf5_path = data_dir / f"episode{episode_idx}.hdf5"
        
        if not hdf5_path.exists():
            print(f"⚠️  Episode {episode_idx} 不存在，跳过")
            continue
        
        # 为每个episode创建子目录
        episode_pc_dir = output_pc_dir / f"episode_{episode_idx}"
        episode_rgb_dir = output_rgb_dir / f"episode_{episode_idx}"
        episode_pc_dir.mkdir(exist_ok=True)
        episode_rgb_dir.mkdir(exist_ok=True)
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # 检查点云数据是否存在
                if 'pointcloud' not in f:
                    print(f"⚠️  Episode {episode_idx} 无点云数据")
                    continue
                
                pointcloud_data = f['pointcloud'][:]  # (T, N, 6)
                
                # 检查RGB数据 - 正确的路径是 observation/{camera_name}/rgb
                rgb_key = f'observation/{camera_name}/rgb'
                if rgb_key not in f:
                    print(f"⚠️  Episode {episode_idx} 无 {camera_name} RGB数据")
                    continue
                
                rgb_data = f[rgb_key][:]  # (T, H, W, 3)
                
                num_frames = pointcloud_data.shape[0]
                
                # 🔧 找到第一个空点云的位置
                first_empty_idx = None
                for frame_idx in range(num_frames):
                    pcd = pointcloud_data[frame_idx]
                    # 检查是否为空点云
                    if np.all(pcd == 0):
                        first_empty_idx = frame_idx
                        break
                    # 检查有效点数量
                    valid_mask = np.any(pcd[:, :3] != 0, axis=1)
                    if np.sum(valid_mask) == 0:
                        first_empty_idx = frame_idx
                        break
                
                # 确定要保存的帧数范围
                if first_empty_idx is not None:
                    max_frames = first_empty_idx
                    truncated_episodes.append({
                        'episode': episode_idx,
                        'original_frames': num_frames,
                        'saved_frames': max_frames,
                        'truncated': num_frames - max_frames
                    })
                else:
                    max_frames = num_frames
                
                # 保存有效的帧
                saved_count = 0
                for frame_idx in range(max_frames):
                    pcd = pointcloud_data[frame_idx]  # (N, 6)
                    rgb = rgb_data[frame_idx]  # (H, W, 3)
                    
                    # 过滤零点（填充点）
                    valid_mask = np.any(pcd[:, :3] != 0, axis=1)
                    pcd_valid = pcd[valid_mask]
                    
                    if len(pcd_valid) == 0:
                        continue
                    
                    # 保存为PLY - 使用process_data_ply.py期望的命名格式
                    output_ply_path = episode_pc_dir / f"step_{frame_idx:04d}.ply"
                    save_ply(pcd_valid, output_ply_path)
                    
                    # 保存RGB图像
                    output_rgb_path = episode_rgb_dir / f"step_{frame_idx:04d}.png"
                    save_rgb(rgb, output_rgb_path)
                    
                    saved_count += 1
                
                total_frames += saved_count
                if first_empty_idx is not None:
                    empty_count += (num_frames - max_frames)
        
        except Exception as e:
            print(f"❌ 处理 Episode {episode_idx} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print(f"✅ 处理完成！")
    print(f"📊 总有效帧数: {total_frames}")
    print(f"⚠️  截断帧数: {empty_count}")
    
    if truncated_episodes:
        print(f"\n� 截断的Episodes ({len(truncated_episodes)}个):")
        for info in truncated_episodes[:10]:  # 只显示前10个
            print(f"  Episode {info['episode']}: {info['original_frames']}帧 → {info['saved_frames']}帧 "
                  f"(截断{info['truncated']}帧)")
        if len(truncated_episodes) > 10:
            print(f"  ... 还有 {len(truncated_episodes) - 10} 个episodes被截断")
    
    print(f"\n�📂 点云输出: {output_pc_dir}")
    print(f"📂 RGB输出: {output_rgb_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="从HDF5提取Sapien原生点云并保存为PLY格式，同时提取RGB")
    parser.add_argument("task_name", type=str, help="任务名称 (e.g., dump_bin_bigbin)")
    parser.add_argument("tag", type=str, help="数据标签 (e.g., demo_randomized)")
    parser.add_argument("num_episodes", type=int, help="要处理的episode数量")
    parser.add_argument("--output_root", type=str, default="./rgbpc_dataset",
                        help="输出根目录 (默认: ./rgbpc_dataset)")
    parser.add_argument("--camera", type=str, default="head_camera",
                        help="相机名称 (默认: head_camera)")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🎯 从HDF5提取Sapien原生点云 + RGB")
    print("=" * 60)
    print(f"任务: {args.task_name}")
    print(f"标签: {args.tag}")
    print(f"Episodes: {args.num_episodes}")
    print(f"相机: {args.camera}")
    print(f"输出根目录: {args.output_root}")
    print("=" * 60 + "\n")
    
    process_sapien_pointcloud(
        task_name=args.task_name,
        tag=args.tag,
        num_episodes=args.num_episodes,
        output_root=args.output_root,
        camera_name=args.camera
    )


if __name__ == "__main__":
    main()
