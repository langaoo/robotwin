import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py
import open3d as o3d

# Try to import pytorch3d for FPS
try:
    import torch
    import pytorch3d.ops as torch3d_ops
    HAS_PYTORCH3D = True
except:
    HAS_PYTORCH3D = False
    print("Warning: pytorch3d not available, FPS sampling will not be available")

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        vector = root["/joint_action/vector"][()]
        # pointcloud is not loaded from HDF5 in this version
        # pointcloud = root["/pointcloud"][()]

    return left_gripper, left_arm, right_gripper, right_arm, vector

def fps_sampling(points, num_points=1024, use_cuda=True):
    """Farthest Point Sampling using pytorch3d"""
    if not HAS_PYTORCH3D:
        raise RuntimeError("pytorch3d not available for FPS sampling")
    
    K = [num_points]
    if use_cuda and torch.cuda.is_available():
        points_tensor = torch.from_numpy(points[:, :3]).float().cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points_tensor.unsqueeze(0), K=K
        )
        indices = indices.squeeze(0).cpu().numpy()
    else:
        points_tensor = torch.from_numpy(points[:, :3]).float()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points_tensor.unsqueeze(0), K=K
        )
        indices = indices.squeeze(0).numpy()
    
    return indices

def load_ply(ply_path, num_points=1024, use_fps=False):
    if not os.path.exists(ply_path):
        print(f"PLY file not found: {ply_path}")
        # Return zeros or handle error? 
        # Better to raise error to avoid bad data
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Open3D reads PLY colors as RGB (0-1 float)
    
    if len(points) == 0:
         return np.zeros((num_points, 6), dtype=np.float32)

    # Concatenate for processing: [XYZ, RGB]
    pointcloud_full = np.concatenate([points, colors], axis=1)
    
    # Sampling
    if use_fps and HAS_PYTORCH3D:
        # Use FPS (Farthest Point Sampling)
        if len(points) >= num_points:
            indices = fps_sampling(pointcloud_full, num_points)
        else:
            # Upsample with FPS + random
            indices_fps = fps_sampling(pointcloud_full, len(points))
            indices_random = np.random.choice(len(points), num_points - len(points), replace=True)
            indices = np.concatenate([indices_fps, indices_random])
    else:
        # Use deterministic sampling based on file path hash to ensure consistency
        # This ensures the same PLY file always produces the same sampled points
        path_hash = hash(ply_path) % (2**32)
        rng = np.random.RandomState(path_hash)
        
        if len(points) >= num_points:
            # Use deterministic random choice with fixed seed
            indices = rng.choice(len(points), num_points, replace=False)
        else:
            # Upsample with replacement
            indices = rng.choice(len(points), num_points, replace=True)
    
    # Apply sampling
    # 🔧 关键: 保持格式为 [XYZ, RGB] 与HDF5原始格式一致
    pointcloud = pointcloud_full[indices]
    
    return pointcloud.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Process some episodes with PLY point clouds.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--ply_dir",
        type=str,
        required=True,
        help="Directory containing PLY files",
    )
    parser.add_argument(
        "--use_fps",
        action="store_true",
        help="Use Farthest Point Sampling (FPS) instead of random sampling",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config
    ply_dir = args.ply_dir

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)

    total_count = 0

    # Save to a different name to distinguish
    suffix = "fps" if args.use_fps else "ply"
    save_dir = f"./data/{task_name}-{task_config}-{num}-{suffix}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    point_cloud_arrays = []
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = (
        [],
        [],
        [],
        [],
    )

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
        ) = load_hdf5(load_path)

        # Determine episode folder in ply_dir
        # Based on ls output: episode_0, episode_1, ...
        # But wait, the ls output showed:
        # policy/DP3/rgbpc_dataset/PC/dump_bin_bigbin-demo_randomized-20_head_camera/episode_4
        # So it is episode_{current_ep}
        
        episode_ply_dir = os.path.join(ply_dir, f"episode_{current_ep}")

        # 正确的时序对齐逻辑：
        # - 时刻 t 的观察: point_cloud[t], state[t]
        # - 从时刻 t 到 t+1 的动作: action[t] = state[t+1] (目标状态)
        # 所以对于 T 个时刻，我们有 T-1 个 (observation, action) 对
        
        ep_data_count = 0  # 记录当前episode实际添加的数据数量
        for j in range(0, left_gripper_all.shape[0] - 1):
            # 时刻 j 的状态和观察
            current_state = vector_all[j]
            # 时刻 j+1 的状态作为目标动作
            next_state = vector_all[j + 1]
            
            # Load PLY for current step j
            ply_filename = f"step_{j:04d}.ply"
            ply_path = os.path.join(episode_ply_dir, ply_filename)
            
            # 🔧 修复：如果PLY文件不存在，跳过这个step
            if not os.path.exists(ply_path):
                # print(f"  警告: PLY文件不存在，跳过 {ply_filename}")
                continue
            
            try:
                pointcloud = load_ply(ply_path, use_fps=args.use_fps)
            except Exception as e:
                print(f"  错误: 加载PLY失败 {ply_filename}: {e}")
                continue
            
            point_cloud_arrays.append(pointcloud)
            state_arrays.append(current_state)
            joint_action_arrays.append(next_state)
            ep_data_count += 1

        current_ep += 1
        total_count += ep_data_count  # 使用实际添加的数据数量
        episode_ends_arrays.append(total_count)

    print()
    try:
        episode_ends_arrays = np.array(episode_ends_arrays)
        state_arrays = np.array(state_arrays)
        point_cloud_arrays = np.array(point_cloud_arrays)
        joint_action_arrays = np.array(joint_action_arrays)
    
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        state_chunk_size = (100, state_arrays.shape[1])
        joint_chunk_size = (100, joint_action_arrays.shape[1])
        point_cloud_chunk_size = (100, point_cloud_arrays.shape[1])
        zarr_data.create_dataset(
            "point_cloud",
            data=point_cloud_arrays,
            chunks=point_cloud_chunk_size,
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "state",
            data=state_arrays,
            chunks=state_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_data.create_dataset(
            "action",
            data=joint_action_arrays,
            chunks=joint_chunk_size,
            dtype="float32",
            overwrite=True,
            compressor=compressor,
        )
        zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends_arrays,
            dtype="int64",
            overwrite=True,
            compressor=compressor,
        )
    except ZeroDivisionError as e:
        print("If you get a `ZeroDivisionError: division by zero`, check that `data/pointcloud` in the task config is set to true.")
        raise 
    except Exception as e:
        print(f"An unexpected error occurred ({type(e).__name__}): {e}")
        raise

if __name__ == "__main__":
    main()
