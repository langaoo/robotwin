import os
import numpy as np
import matplotlib.pyplot as plt
from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset

# 创建保存图像的目录
save_dir = "augmentation_verify"
os.makedirs(save_dir, exist_ok=True)

# 1. 初始化两个数据集（使用不同的配置文件：一个有增强，一个无增强）
augmented_dataset = RobotImageDataset(
    zarr_path="data/beat_block_hammer-demo_randomized-50_multi_cam.zarr",
    task_config_path="/home/gl/RoboTwin/task_config/demo_augmented.yml",
    horizon=1,
    val_ratio=0.0
)

original_dataset = RobotImageDataset(
    zarr_path="data/beat_block_hammer-demo_randomized-50_multi_cam.zarr",
    task_config_path="/home/gl/RoboTwin/task_config/demo_randomized.yml",
    horizon=1,
    val_ratio=0.0
)

# 2. 选择样本
sample_idx = 0
print(f"正在加载样本 {sample_idx}...")
augmented_sample = augmented_dataset[sample_idx]
original_sample = original_dataset[sample_idx]

# 3. 可视化所有四个摄像头
camera_names = ['head_cam', 'front_cam', 'left_cam', 'right_cam']
camera_titles = ['Head Camera', 'Front Camera', 'Left Camera', 'Right Camera']

fig, axes = plt.subplots(4, 2, figsize=(12, 16))
fig.suptitle(f'Sample {sample_idx}: Original vs Augmented (4 Cameras)', fontsize=16)

for i, (cam_name, cam_title) in enumerate(zip(camera_names, camera_titles)):
    aug_img = augmented_sample["obs"][cam_name][0].numpy()
    orig_img = original_sample["obs"][cam_name][0].numpy()
    
    aug_img = np.transpose(aug_img, (1, 2, 0)) * 255
    orig_img = np.transpose(orig_img, (1, 2, 0)) * 255
    aug_img = aug_img.astype(np.uint8)
    orig_img = orig_img.astype(np.uint8)
    
    axes[i, 0].imshow(orig_img)
    axes[i, 0].set_title(f'{cam_title} - Original')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(aug_img)
    axes[i, 1].set_title(f'{cam_title} - Augmented')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "all_cameras_comparison.png"), dpi=150)
plt.close()

print(f"✓ 所有摄像头对比图已保存到：{os.path.join(save_dir, 'all_cameras_comparison.png')}")

# 4. 生成多个随机增强版本（验证随机性）
print("\n正在生成多个增强样本以验证随机性...")
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle(f'Head Camera - Sample {sample_idx}: Multiple Augmented Versions', fontsize=16)

orig_img = original_sample["obs"]["head_cam"][0].numpy()
orig_img = np.transpose(orig_img, (1, 2, 0)) * 255
axes[0, 0].imshow(orig_img.astype(np.uint8))
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

for idx in range(1, 9):
    aug_sample = augmented_dataset[sample_idx]
    aug_img = aug_sample["obs"]["head_cam"][0].numpy()
    aug_img = np.transpose(aug_img, (1, 2, 0)) * 255
    
    row = idx // 3
    col = idx % 3
    axes[row, col].imshow(aug_img.astype(np.uint8))
    axes[row, col].set_title(f'Augmented #{idx}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "random_augmentation_variations.png"), dpi=150)
plt.close()

print(f"✓ 随机增强变化对比图已保存到：{os.path.join(save_dir, 'random_augmentation_variations.png')}")
