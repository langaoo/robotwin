import numpy as np
import torch
from torchvision import transforms
import cv2  
from diffusion_policy.common.pytorch_util import dict_apply

# 1. 物体姿态随机化（假设state中包含物体位置，通过旋转矩阵扰动）
def augment_object_pose(self, obs):
    # 假设agent_pos的后3维为物体位置(x,y,theta)，对theta添加旋转扰动
    agent_pos = obs['agent_pos']
    if agent_pos.shape[-1] >= 3:  # 确保包含姿态信息
        rot_range = self.object_pose_rot_range
        rot_delta = np.random.uniform(rot_range[0], rot_range[1]) * np.pi / 180  # 转为弧度
        agent_pos[..., -1] += rot_delta  # 扰动角度
        agent_pos[..., -1] = (agent_pos[..., -1] + np.pi) % (2 * np.pi) - np.pi  # 归一化到[-π, π]
    return obs

# 2. 相机视角抖动（对图像进行随机旋转/平移模拟视角变化）
def augment_camera_jitter(self, obs):
    head_cam = obs['head_cam']  # 形状：(T, 3, H, W)
    T, C, H, W = head_cam.shape
    jittered = []
    for t in range(T):
        img = head_cam[t]  # (3, H, W)
        # 随机旋转（-5~5度）
        angle = np.random.uniform(self.camera_jitter_range[0], self.camera_jitter_range[1])
        img_np = img.permute(1, 2, 0).numpy()  # (H, W, 3)
        rows, cols = img_np.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(img_np, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        jittered.append(torch.from_numpy(rotated).permute(2, 0, 1))  # 转回(C, H, W)
    obs['head_cam'] = torch.stack(jittered)
    return obs

# 3. 传感器噪声注入（对state添加高斯噪声）
def augment_sensor_noise(self, obs):
    agent_pos = obs['agent_pos']
    signal_power = np.var(agent_pos.numpy()) if isinstance(agent_pos, torch.Tensor) else np.var(agent_pos)
    noise_power = signal_power / (10 **(self.sensor_noise_snr / 10))
    noise = torch.normal(0, np.sqrt(noise_power), size=agent_pos.shape, device=agent_pos.device)
    obs['agent_pos'] += noise
    return obs

# 4. 样本混合（mix-up，混合两个样本的观测和动作）
def augment_mixup(self, data):
    # 随机选择另一个样本索引
    other_idx = np.random.randint(0, len(self))
    other_sample = self.sampler.sample_sequence(other_idx)
    other_data = self._sample_to_data(other_sample)
    other_data = dict_apply(other_data, torch.from_numpy)
    
    # 生成mixup系数
    lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
    
    # 混合观测和动作
    mixed_data = {}
    mixed_data['obs'] = {}
    mixed_data['obs']['head_cam'] = lam * data['obs']['head_cam'] + (1 - lam) * other_data['obs']['head_cam']
    mixed_data['obs']['agent_pos'] = lam * data['obs']['agent_pos'] + (1 - lam) * other_data['obs']['agent_pos']
    mixed_data['action'] = lam * data['action'] + (1 - lam) * other_data['action']
    return mixed_data

# 5. 颜色抖动（调整亮度、对比度、饱和度）
def augment_color_jitter(self, obs):
    head_cam = obs['head_cam']  # (T, 3, H, W)
    T, C, H, W = head_cam.shape
    jittered = []
    for t in range(T):
        img = head_cam[t]  # (3, H, W)
        jittered_img = self.color_jitter(img)
        jittered.append(jittered_img)
    obs['head_cam'] = torch.stack(jittered)
    return obs
