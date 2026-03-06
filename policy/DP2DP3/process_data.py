"""
DP2DP3 数据处理脚本 — 从 RoboTwin HDF5 生成 FiLM 训练所需的 zarr 数据集.

与 DP/process_data.py 的区别:
  - 只提取 head_camera（FiLM 只用一个视角），节省磁盘和时间
  - 输出路径在 DP2DP3/data/ 下，与 DP 解耦
  - 支持 --cameras 参数提取额外摄像头（为将来多摄像头实验预留）

输出 zarr 格式 (与 DP 兼容):
  data/head_camera:  [N, 3, H, W] uint8  (NCHW)
  data/state:        [N, 14]      float32
  data/action:       [N, 14]      float32
  meta/episode_ends: [n_ep]       int64

用法:
  cd /home/gl/RoboTwin/policy/DP2DP3
  python process_data.py beat_block_hammer demo_clean 50
  python process_data.py move_can_pot demo_clean 50
  python process_data.py lift_pot demo_clean 50

  # 提取多个摄像头:
  python process_data.py beat_block_hammer demo_clean 50 --cameras head_camera front_camera
"""

import argparse
import os
import shutil
import sys

import cv2
import h5py
import numpy as np
import zarr


def load_hdf5(dataset_path):
    """加载单个 episode 的 HDF5 数据."""
    if not os.path.isfile(dataset_path):
        print(f"HDF5 文件不存在: {dataset_path}")
        sys.exit(1)

    with h5py.File(dataset_path, "r") as root:
        vector = root["/joint_action/vector"][()]

        image_dict = {}
        for cam_name in root["/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return vector, image_dict


def main():
    parser = argparse.ArgumentParser(
        description="DP2DP3 数据处理: HDF5 -> zarr (FiLM 训练用)"
    )
    parser.add_argument("task_name", type=str, help="任务名 (e.g. beat_block_hammer)")
    parser.add_argument("task_config", type=str, help="配置名 (e.g. demo_clean)")
    parser.add_argument("expert_data_num", type=int, help="episode 数 (e.g. 50)")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["head_camera"],
        help="要提取的摄像头列表 (默认: head_camera)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="HDF5 源数据根目录 (默认: ../../data/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="zarr 输出目录 (默认: ./data/)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    num = args.expert_data_num
    cameras = args.cameras

    # 路径设置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data_root:
        load_dir = os.path.join(args.data_root, task_name, task_config)
    else:
        load_dir = os.path.join(script_dir, "..", "..", "data", task_name, task_config)
    load_dir = os.path.normpath(load_dir)

    output_dir = args.output_dir or os.path.join(script_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.join(output_dir, f"{task_name}-{task_config}-{num}_multi_cam.zarr")

    print(f"任务: {task_name}")
    print(f"配置: {task_config}")
    print(f"Episodes: {num}")
    print(f"摄像头: {cameras}")
    print(f"HDF5 源目录: {load_dir}")
    print(f"zarr 输出: {save_dir}")

    if not os.path.isdir(load_dir):
        print(f"\n错误: HDF5 源目录不存在: {load_dir}")
        sys.exit(1)

    # 清理已有 zarr
    if os.path.exists(save_dir):
        print(f"\n已存在 zarr 将被覆盖: {save_dir}")
        shutil.rmtree(save_dir)

    # 初始化 zarr
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # 收集数据
    camera_arrays = {cam: [] for cam in cameras}
    state_arrays = []
    action_arrays = []
    episode_ends = []
    total_count = 0

    for ep in range(num):
        load_path = os.path.join(load_dir, "data", f"episode{ep}.hdf5")
        print(f"  处理 episode {ep + 1}/{num}", end="\r")

        vector, image_dict = load_hdf5(load_path)
        n_frames = vector.shape[0]

        for j in range(n_frames):
            joint_state = vector[j]

            # 非最后一帧: 记录 state + 图像
            if j != n_frames - 1:
                for cam_name in cameras:
                    if cam_name in image_dict:
                        img_bit = image_dict[cam_name][j]
                        img = cv2.imdecode(
                            np.frombuffer(img_bit, np.uint8), cv2.IMREAD_COLOR
                        )
                        camera_arrays[cam_name].append(img)
                    else:
                        print(
                            f"\n警告: episode {ep} 中未找到摄像头 {cam_name}, 跳过"
                        )

                state_arrays.append(joint_state)

            # 非第一帧: 记录 action
            if j != 0:
                action_arrays.append(joint_state)

        total_count += n_frames - 1
        episode_ends.append(total_count)

    print(f"\n\n总帧数: {total_count}, episodes: {num}")

    # 保存到 zarr
    print("保存到 zarr ...")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    # state
    state_np = np.array(state_arrays, dtype=np.float32)
    zarr_data.create_dataset(
        "state",
        data=state_np,
        chunks=(100, state_np.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    print(f"  state: {state_np.shape}")

    # action
    action_np = np.array(action_arrays, dtype=np.float32)
    zarr_data.create_dataset(
        "action",
        data=action_np,
        chunks=(100, action_np.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    print(f"  action: {action_np.shape}")

    # episode_ends
    ends_np = np.array(episode_ends, dtype=np.int64)
    zarr_meta.create_dataset(
        "episode_ends",
        data=ends_np,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )
    print(f"  episode_ends: {ends_np.shape}")

    # 摄像头图像
    for cam_name, cam_data in camera_arrays.items():
        if len(cam_data) == 0:
            print(f"  警告: {cam_name} 无数据, 跳过")
            continue

        cam_np = np.array(cam_data)
        cam_np = np.moveaxis(cam_np, -1, 1)  # NHWC -> NCHW

        zarr_data.create_dataset(
            cam_name,
            data=cam_np,
            chunks=(100, *cam_np.shape[1:]),
            overwrite=True,
            compressor=compressor,
        )
        print(f"  {cam_name}: {cam_np.shape}")

    print(f"\n完成! zarr 已保存到: {save_dir}")


if __name__ == "__main__":
    main()
