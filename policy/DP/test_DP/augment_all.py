import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/gl/RoboTwin/policy/DP')
from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset

save_dir = "all_augmentations_verify"
os.makedirs(save_dir, exist_ok=True)

print("=" * 70)
print("测试所有数据增强方案")
print("=" * 70)

# 加载完整增强的数据集
print("\n[1] 加载完整增强配置的数据集...")
full_aug_dataset = RobotImageDataset(
    zarr_path="data/beat_block_hammer-demo_randomized-50_multi_cam.zarr",
    task_config_path="/home/gl/RoboTwin/task_config/demo_augmented_full.yml",
    horizon=1,
    val_ratio=0.0
)

# 加载无增强的数据集
print("[2] 加载无增强配置的数据集...")
no_aug_dataset = RobotImageDataset(
    zarr_path="data/beat_block_hammer-demo_randomized-50_multi_cam.zarr",
    task_config_path="/home/gl/RoboTwin/task_config/demo_clean.yml",
    horizon=1,
    val_ratio=0.0
)

sample_idx = 0
print(f"\n[3] 测试样本 #{sample_idx}")

# 获取原始样本
original = no_aug_dataset[sample_idx]

# ========== 测试1: 四个摄像头的增强效果 ==========
print("\n[4] 生成四摄像头增强对比图...")
camera_names = ['head_cam', 'front_cam', 'left_cam', 'right_cam']
camera_titles = ['Head Camera', 'Front Camera', 'Left Camera', 'Right Camera']

fig, axes = plt.subplots(4, 4, figsize=(18, 18))
fig.suptitle(f'All Augmentations: Sample #{sample_idx} (4 Cameras)', fontsize=16, y=0.995)

for i, (cam_name, cam_title) in enumerate(zip(camera_names, camera_titles)):
    # 原始图像
    orig_img = original["obs"][cam_name][0].numpy()
    orig_img = np.transpose(orig_img, (1, 2, 0)) * 255
    orig_img = orig_img.astype(np.uint8)
    
    axes[i, 0].imshow(orig_img)
    axes[i, 0].set_title(f'{cam_title}\nOriginal', fontsize=10)
    axes[i, 0].axis('off')
    
    # 三个不同的增强版本
    for j in range(1, 4):
        aug_sample = full_aug_dataset[sample_idx]
        aug_img = aug_sample["obs"][cam_name][0].numpy()
        aug_img = np.transpose(aug_img, (1, 2, 0)) * 255
        aug_img = aug_img.astype(np.uint8)
        
        axes[i, j].imshow(aug_img)
        axes[i, j].set_title(f'{cam_title}\nAugmented #{j}', fontsize=10)
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "all_aug_4cameras.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"    ✓ 保存到: {os.path.join(save_dir, 'all_aug_4cameras.png')}")

# ========== 测试2: Head Camera的多样性展示 ==========
print("\n[5] 生成Head Camera增强多样性展示...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle(f'Head Camera: Augmentation Diversity (Sample #{sample_idx})', fontsize=16)

# 原始图像
orig_img = original["obs"]["head_cam"][0].numpy()
orig_img = np.transpose(orig_img, (1, 2, 0)) * 255
orig_img = orig_img.astype(np.uint8)
axes[0, 0].imshow(orig_img)
axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# 11个不同的增强版本
for idx in range(1, 12):
    aug_sample = full_aug_dataset[sample_idx]
    aug_img = aug_sample["obs"]["head_cam"][0].numpy()
    aug_img = np.transpose(aug_img, (1, 2, 0)) * 255
    aug_img = aug_img.astype(np.uint8)
    
    row = idx // 4
    col = idx % 4
    axes[row, col].imshow(aug_img)
    axes[row, col].set_title(f'Augmented #{idx}', fontsize=10)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "head_cam_diversity.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"    ✓ 保存到: {os.path.join(save_dir, 'head_cam_diversity.png')}")

# ========== 测试3: 不同增强方案的单独测试 ==========
print("\n[6] 测试单独增强方案的效果...")

augmentation_configs = {
    'Camera Jitter': {
        'use_camera_jitter': True,
        'camera_jitter': {'translate_percent': 0.05, 'scale': [0.95, 1.05], 'shear': [-5, 5], 'prob': 1.0}
    },
    'Perspective': {
        'use_perspective': True,
        'perspective': {'scale': [0.05, 0.15], 'prob': 1.0}
    },
    'Lighting': {
        'use_lighting': True,
        'lighting': {'brightness_range': [0.7, 1.3], 'shadow_prob': 0.5, 'highlight_prob': 0.5}
    },
    'Occlusion': {
        'use_occlusion': True,
        'occlusion': {'method': 'random_erasing', 'prob': 1.0, 'scale': [0.02, 0.15], 'ratio': [0.3, 3.3]}
    },
    'Background Blur': {
        'use_background_blur': True,
        'background_blur': {'prob': 1.0, 'sigma': [1.0, 3.0]}
    },
    'Mix-up': {
        'use_mixup': True,
        'mixup': {'prob': 1.0, 'alpha': 0.4}
    }
}

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(f'Individual Augmentation Effects (Head Camera, Sample #{sample_idx})', fontsize=16)

# 原始图像
axes[0, 0].imshow(orig_img)
axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# 逐个测试增强方案
for idx, (aug_name, aug_cfg) in enumerate(augmentation_configs.items(), 1):
    print(f"    测试 {aug_name}...")
    
    # 创建临时配置
    temp_config_path = f"/tmp/temp_aug_{idx}.yml"
    with open(temp_config_path, 'w') as f:
        yaml.dump({'data_augmentation': aug_cfg}, f)
    
    # 加载数据集
    temp_dataset = RobotImageDataset(
        zarr_path="data/beat_block_hammer-demo_randomized-50_multi_cam.zarr",
        task_config_path=temp_config_path,
        horizon=1,
        val_ratio=0.0
    )
    
    # 获取增强样本
    temp_sample = temp_dataset[sample_idx]
    temp_img = temp_sample["obs"]["head_cam"][0].numpy()
    temp_img = np.transpose(temp_img, (1, 2, 0)) * 255
    temp_img = temp_img.astype(np.uint8)
    
    row = idx // 4
    col = idx % 4
    axes[row, col].imshow(temp_img)
    axes[row, col].set_title(aug_name, fontsize=11)
    axes[row, col].axis('off')
    
    # 清理临时文件
    os.remove(temp_config_path)

# 隐藏空白子图
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "individual_augmentations.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"    ✓ 保存到: {os.path.join(save_dir, 'individual_augmentations.png')}")

# ========== 总结 ==========
print("\n" + "=" * 70)
print("测试完成！生成的文件:")
print(f"  1. {os.path.join(save_dir, 'all_aug_4cameras.png')}")
print(f"     → 四个摄像头的完整增强对比")
print(f"  2. {os.path.join(save_dir, 'head_cam_diversity.png')}")
print(f"     → Head Camera的增强多样性展示")
print(f"  3. {os.path.join(save_dir, 'individual_augmentations.png')}")
print(f"     → 各个增强方案的单独效果")
print("=" * 70)

print("\n📌 增强方案列表:")
print("  ✓ 随机裁剪 (Random Crop)")
print("  ✓ 随机旋转 (Random Rotation)")
print("  ✓ 颜色抖动 (Color Jitter)")
print("  ✓ 传感器噪声 (Sensor Noise)")
print("  ✓ 相机视角抖动 (Camera Jitter)")
print("  ✓ 透视变换 (Perspective Transform)")
print("  ✓ 光照变化 (Lighting Augmentation)")
print("  ✓ 遮挡模拟 (Occlusion)")
print("  ✓ 背景模糊 (Background Blur)")
print("  ✓ Mix-up")