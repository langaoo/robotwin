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


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # 加载关节动作数据
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        vector = root["/joint_action/vector"][()]
        
        # 加载所有摄像头图像数据（支持多摄像头）
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            # 存储每个摄像头的所有帧数据
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, vector, image_dict


def main():
    parser = argparse.ArgumentParser(description="Process some episodes with multiple cameras.")
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
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)
    save_dir = f"./data/{task_name}-{task_config}-{num}_multi_cam.zarr"

    # 清理已有文件
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # 初始化Zarr存储
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # 存储所有摄像头的图像数据（动态字典，自动适配存在的摄像头）
    camera_arrays = {
        "head_camera": [],
        "front_camera": [],
        "left_camera": [],
        "right_camera": []
    }
    episode_ends_arrays = []
    state_arrays = []
    joint_action_arrays = []
    total_count = 0
    current_ep = 0

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")
        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        
        # 加载当前episode的所有数据
        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            image_dict_all,
        ) = load_hdf5(load_path)

        # 遍历当前episode的所有帧
        for j in range(left_gripper_all.shape[0]):
            # 提取关节状态
            joint_state = vector_all[j]

            # 处理图像数据（仅非最后一帧，与状态对齐）
            if j != left_gripper_all.shape[0] - 1:
                # 遍历所有需要处理的摄像头
                for cam_name in camera_arrays.keys():
                    # 检查当前摄像头是否存在于数据中
                    if cam_name in image_dict_all:
                        # 解码图像（HDF5中存储的是压缩字节流）
                        img_bit = image_dict_all[cam_name][j]
                        img = cv2.imdecode(np.frombuffer(img_bit, np.uint8), cv2.IMREAD_COLOR)
                        camera_arrays[cam_name].append(img)
                
                # 存储关节状态
                state_arrays.append(joint_state)
            
            # 处理动作数据（仅非第一帧，与前一状态对齐）
            if j != 0:
                joint_action_arrays.append(joint_state)

        # 更新episode结束标记
        current_ep += 1
        total_count += left_gripper_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

    print("\nSaving data to Zarr...")
    # 转换为numpy数组
    episode_ends_arrays = np.array(episode_ends_arrays)
    state_arrays = np.array(state_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    
    # 配置压缩器
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    # 保存关节状态和动作数据
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=(100, state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=joint_action_arrays,
        chunks=(100, joint_action_arrays.shape[1]),
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

    # 保存所有摄像头的图像数据（仅保存存在数据的摄像头）
    for cam_name, cam_data in camera_arrays.items():
        if len(cam_data) == 0:
            print(f"Warning: No data found for {cam_name}, skipping...")
            continue
        
        # 转换为NCHW格式（适配多数视觉模型输入）
        cam_array = np.array(cam_data)
        cam_array = np.moveaxis(cam_array, -1, 1)  # NHWC -> NCHW
        
        # 保存到Zarr
        zarr_data.create_dataset(
            cam_name,
            data=cam_array,
            chunks=(100, *cam_array.shape[1:]),
            overwrite=True,
            compressor=compressor,
        )
        print(f"Saved {cam_name} with shape: {cam_array.shape}")

    print("Processing complete!")


if __name__ == "__main__":
    main()