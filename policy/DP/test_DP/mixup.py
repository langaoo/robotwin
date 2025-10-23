import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加路径
sys.path.append('/home/gl/RoboTwin/policy/DP')
from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset

# 创建保存图像的目录
save_dir = "mixup_verify"
os.makedirs(save_dir, exist_ok=True)

print("=" * 60)
print("Mix-up 数据增强测试")
print("=" * 60)

# 1. 创建启用Mix-up的数据集
print("\n[1] 加载启用Mix-up的数据集...")
mixup_dataset = RobotImageDataset(
    zarr_path="data/beat_block_hammer-demo_randomized-50_multi_cam.zarr",
    task_config_path="/home/gl/RoboTwin/task_config/demo_augmented.yml",
    horizon=1,
    val_ratio=0.0
)

# 2. 创建不启用Mix-up的数据集（用于对比）
print("[2] 加载不启用Mix-up的数据集...")
no_mixup_dataset = RobotImageDataset(
    zarr_path="data/beat_block_hammer-demo_randomized-50_multi_cam.zarr",
    task_config_path="/home/gl/RoboTwin/task_config/demo_randomized.yml",
    horizon=1,
    val_ratio=0.0
)

# 3. 选择两个不同的样本进行测试
sample_idx_1 = 0
sample_idx_2 = 5

print(f"\n[3] 测试样本: #{sample_idx_1} 和 #{sample_idx_2}")

# 获取原始样本（无Mix-up）
original_sample_1 = no_mixup_dataset[sample_idx_1]
original_sample_2 = no_mixup_dataset[sample_idx_2]

print(f"    原始样本1的动作: {original_sample_1['action'][0].numpy()[:3]}...")
print(f"    原始样本2的动作: {original_sample_2['action'][0].numpy()[:3]}...")

# 4. 可视化: 显示Mix-up前后的对比（所有4个摄像头）
print("\n[4] 生成Mix-up对比图（4个摄像头）...")

camera_names = ['head_cam', 'front_cam', 'left_cam', 'right_cam']
camera_titles = ['Head Camera', 'Front Camera', 'Left Camera', 'Right Camera']

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle(f'Mix-up Effect: Sample #{sample_idx_1} + Sample #{sample_idx_2}', fontsize=16)

for i, (cam_name, cam_title) in enumerate(zip(camera_names, camera_titles)):
    # 原始样本1
    img1 = original_sample_1["obs"][cam_name][0].numpy()
    img1 = np.transpose(img1, (1, 2, 0)) * 255
    img1 = img1.astype(np.uint8)
    
    # 原始样本2
    img2 = original_sample_2["obs"][cam_name][0].numpy()
    img2 = np.transpose(img2, (1, 2, 0)) * 255
    img2 = img2.astype(np.uint8)
    
    # Mix-up结果（多次采样）
    mixup_sample = mixup_dataset[sample_idx_1]
    mixup_img = mixup_sample["obs"][cam_name][0].numpy()
    mixup_img = np.transpose(mixup_img, (1, 2, 0)) * 255
    mixup_img = mixup_img.astype(np.uint8)
    
    # 手动Mix-up参考（lambda=0.5）
    manual_mixup = (img1.astype(np.float32) * 0.5 + img2.astype(np.float32) * 0.5).astype(np.uint8)
    
    # 显示
    axes[i, 0].imshow(img1)
    axes[i, 0].set_title(f'{cam_title}\nOriginal Sample #{sample_idx_1}')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(img2)
    axes[i, 1].set_title(f'{cam_title}\nOriginal Sample #{sample_idx_2}')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(mixup_img)
    axes[i, 2].set_title(f'{cam_title}\nMix-up Result (Random λ)')
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(manual_mixup)
    axes[i, 3].set_title(f'{cam_title}\nReference (λ=0.5)')
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "mixup_4cameras_comparison.png"), dpi=150)
plt.close()

print(f"    ✓ 保存到: {os.path.join(save_dir, 'mixup_4cameras_comparison.png')}")

# 5. 生成多个不同lambda值的Mix-up版本（只看head_cam）
print("\n[5] 生成不同Mix-up强度的对比图（Head Camera）...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f'Head Camera: Different Mix-up Versions (Sample #{sample_idx_1})', fontsize=16)

# 原始样本
img1 = original_sample_1["obs"]["head_cam"][0].numpy()
img1 = np.transpose(img1, (1, 2, 0)) * 255
img1 = img1.astype(np.uint8)

axes[0, 0].imshow(img1)
axes[0, 0].set_title(f'Original Sample #{sample_idx_1}')
axes[0, 0].axis('off')

# 多次采样Mix-up结果（验证随机性）
for idx in range(1, 8):
    mixup_sample = mixup_dataset[sample_idx_1]
    mixup_img = mixup_sample["obs"]["head_cam"][0].numpy()
    mixup_img = np.transpose(mixup_img, (1, 2, 0)) * 255
    mixup_img = mixup_img.astype(np.uint8)
    
    row = idx // 4
    col = idx % 4
    axes[row, col].imshow(mixup_img)
    axes[row, col].set_title(f'Mix-up Version #{idx}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "mixup_random_variations.png"), dpi=150)
plt.close()

print(f"    ✓ 保存到: {os.path.join(save_dir, 'mixup_random_variations.png')}")

# 6. 验证Mix-up对动作的影响
print("\n[6] 验证Mix-up对动作空间的影响...")

# 获取多个Mix-up样本，检查动作是否也被混合
print("\n    采样10次，查看动作混合情况:")
for i in range(10):
    mixup_sample = mixup_dataset[sample_idx_1]
    action = mixup_sample['action'][0].numpy()
    print(f"    Mix-up #{i+1}: action[:3] = {action[:3]}")

print("\n    参考原始动作:")
print(f"    样本#{sample_idx_1}: {original_sample_1['action'][0].numpy()[:3]}")
print(f"    样本#{sample_idx_2}: {original_sample_2['action'][0].numpy()[:3]}")

# 7. 总结
print("\n" + "=" * 60)
print("测试完成！生成的文件:")
print(f"  1. {os.path.join(save_dir, 'mixup_4cameras_comparison.png')}")
print(f"     → 显示Mix-up前后对比（4个摄像头）")
print(f"  2. {os.path.join(save_dir, 'mixup_random_variations.png')}")
print(f"     → 显示多个随机Mix-up版本（验证随机性）")
print("=" * 60)

print("\n 如何验证Mix-up是否生效:")
print("  1. 查看图像是否是两个样本的混合（半透明效果）")
print("  2. 检查动作值是否在两个原始样本之间")
print("  3. 多次运行看Mix-up强度是否随机变化")