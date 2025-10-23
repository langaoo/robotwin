from typing import Dict
import numba
import torch
import numpy as np
import copy
from imgaug import augmenters as iaa  # 确保安装imgaug: pip install imgaug
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
import pdb
import cv2
from torchvision import transforms
from omegaconf import OmegaConf
import os
import random
import yaml



class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        task_config_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
    ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            # keys=["head_camera", "state", "action"],
        )

        # 加载task_config中的增强参数
        self.aug_config = self._load_augmentation_config(task_config_path)

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()
        # 初始化增强变换
        self.img_transforms = self._build_img_transforms()
        self.sensor_noise_aug = self._build_sensor_noise_aug()
        self.camera_jitter_aug = self._build_camera_jitter_aug()
        self.occlusion_aug = self._build_occlusion_aug()

    def _load_augmentation_config(self, config_path):
        """从task_config加载增强参数"""
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Task config file not found: {config_path}")
        
        # 读取YAML配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # 默认增强配置
        default_aug = {
            'use_random_crop': False,
            'crop_size': (224, 224),
            'use_rotation': False,
            'rotation_degrees': (-10, 10),
            'use_color_jitter': False,
            'color_jitter': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.05
            },
            # 传感器噪声
            'use_sensor_noise': False,
            'sensor_noise': {
                'gaussian_scale': 0.01,
                'blur_sigma': (0, 0.5)
            },
            # 相机视角抖动
            'use_camera_jitter': False,
            'camera_jitter': {
                'translate_percent': 0.05,  # 平移范围（相对图像尺寸）
                'scale': (0.95, 1.05),      # 缩放范围
                'shear': (-5, 5),           # 剪切角度
                'prob': 0.5                 # 应用概率
            },
            # 透视变换
            'use_perspective': False,
            'perspective': {
                'scale': (0.05, 0.15),      # 透视变换强度
                'prob': 0.3
            },
            # 遮挡模拟
            'use_occlusion': False,
            'occlusion': {
                'method': 'random_erasing',  # 'random_erasing' or 'cutout'
                'prob': 0.5,
                'scale': (0.02, 0.15),      # 遮挡区域占比
                'ratio': (0.3, 3.3),        # 遮挡区域宽高比
                'num_holes': 1              # 遮挡块数量（cutout方法）
            },
            # Mix-up增强
            'use_mixup': False,
            'mixup': {
                'prob': 0.5,
                'alpha': 0.4
            }
        }
        
        # 合并配置（优先使用文件中的配置）
        aug_config = config.get('data_augmentation', {})
        return {** default_aug, **aug_config}

    def _build_img_transforms(self):
        """构建图像增强变换管道"""
        transforms_list = [transforms.ToPILImage()]  # 转为PIL便于处理

        # 随机裁剪
        if self.aug_config['use_random_crop']:
            crop_size = tuple(self.aug_config['crop_size'])
            transforms_list.append(transforms.RandomResizedCrop(
                size=crop_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ))

        # 随机旋转
        if self.aug_config['use_rotation']:
            degrees = self.aug_config['rotation_degrees']
            transforms_list.append(transforms.RandomRotation(degrees=degrees))

        # 颜色抖动
        if self.aug_config['use_color_jitter']:
            jitter_cfg = self.aug_config['color_jitter']
            transforms_list.append(transforms.ColorJitter(
                brightness=jitter_cfg['brightness'],
                contrast=jitter_cfg['contrast'],
                saturation=jitter_cfg['saturation'],
                hue=jitter_cfg['hue']
            ))

        # 转回Tensor并归一化到0-1
        transforms_list.append(transforms.ToTensor())
        return transforms.Compose(transforms_list)

    def _build_sensor_noise_aug(self):
        """构建传感器噪声增强器"""
        if self.aug_config['use_sensor_noise']:
            noise_cfg = self.aug_config['sensor_noise']
            return iaa.Sequential([
                iaa.AdditiveGaussianNoise(scale=(0, noise_cfg['gaussian_scale'] * 255)),
                iaa.GaussianBlur(sigma=noise_cfg['blur_sigma']),
            ])
        return None
    def _build_camera_jitter_aug(self):
        """构建相机视角抖动增强器"""
        if self.aug_config['use_camera_jitter']:
            jitter_cfg = self.aug_config['camera_jitter']
            aug_list = []
            
            # 仿射变换（平移、缩放、剪切）
            aug_list.append(
                iaa.Sometimes(
                    jitter_cfg['prob'],
                    iaa.Affine(
                        translate_percent={
                            "x": (-jitter_cfg['translate_percent'], jitter_cfg['translate_percent']),
                            "y": (-jitter_cfg['translate_percent'], jitter_cfg['translate_percent'])
                        },
                        scale=jitter_cfg['scale'],
                        shear=jitter_cfg['shear'],
                        mode='edge'  # 边缘填充模式
                    )
                )
            )
            
            # 透视变换
            if self.aug_config['use_perspective']:
                persp_cfg = self.aug_config['perspective']
                aug_list.append(
                    iaa.Sometimes(
                        persp_cfg['prob'],
                        iaa.PerspectiveTransform(scale=persp_cfg['scale'])
                    )
                )
            
            return iaa.Sequential(aug_list)
        return None

    def _build_occlusion_aug(self):
        """构建遮挡模拟增强器"""
        if self.aug_config['use_occlusion']:
            occ_cfg = self.aug_config['occlusion']
            
            if occ_cfg['method'] == 'cutout':
                # Cutout方法：固定大小的黑色方块
                return iaa.Sometimes(
                    occ_cfg['prob'],
                    iaa.Cutout(
                        nb_iterations=occ_cfg['num_holes'],
                        size=0.1,  # 相对大小
                        squared=False
                    )
                )
            else:
                # Random Erasing方法：随机大小和宽高比
                return iaa.Sometimes(
                    occ_cfg['prob'],
                    iaa.CoarseDropout(
                        p=0.2,  # 每个区域被遮挡的概率
                        size_percent=occ_cfg['scale']
                    )
                )
        return None
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        # 强制关闭验证集增强
        val_set.aug_config = {k: (v if not k.startswith('use_') else False) 
                             for k, v in val_set.aug_config.items()}
        val_set.img_transforms = val_set._build_img_transforms()
        val_set.sensor_noise_aug = val_set._build_sensor_noise_aug()
        val_set.camera_jitter_aug = val_set._build_camera_jitter_aug()
        val_set.occlusion_aug = val_set._build_occlusion_aug()
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["front_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)  # (agent_posx2, block_posex3)
        # 原始数据格式：zarr中保存的是(T, C, H, W)格式
        head_cam = sample["head_camera"]
        front_cam = sample['front_camera']
        left_cam = sample['left_camera']
        right_cam = sample['right_camera']

        # 检查是否需要应用增强
        need_aug = any([
            self.aug_config['use_random_crop'],
            self.aug_config['use_rotation'],
            self.aug_config['use_color_jitter'],
            self.aug_config['use_sensor_noise'],
            self.aug_config['use_camera_jitter'],
            self.aug_config['use_perspective'],
            self.aug_config['use_occlusion'],
        ])

        if need_aug:
            T = head_cam.shape[0]  # 时间步长
            # 为每个相机初始化增强后的列表
            augmented_cams = {
                'head': [], 'front': [], 'left': [], 'right': []
            }

            for t in range(T):
                # 处理每个相机的单帧图像
                # 数据是(C, H, W)格式，需要转换为(H, W, C)格式给PIL
                cams = {
                    'head': np.transpose(head_cam[t], (1, 2, 0)).astype(np.uint8),
                    'front': np.transpose(front_cam[t], (1, 2, 0)).astype(np.uint8),
                    'left': np.transpose(left_cam[t], (1, 2, 0)).astype(np.uint8),
                    'right': np.transpose(right_cam[t], (1, 2, 0)).astype(np.uint8)
                }

                # 应用增强（按顺序）
                for cam_name in cams:
                    img = cams[cam_name]
                    
                    # 1. 传感器噪声
                    if self.sensor_noise_aug is not None:
                        img = self.sensor_noise_aug(image=img)
                    
                    # 2. 相机视角抖动 + 透视变换
                    if self.camera_jitter_aug is not None:
                        img = self.camera_jitter_aug(image=img)

                    # 3. 遮挡模拟
                    if self.occlusion_aug is not None:
                        img = self.occlusion_aug(image=img)

                    # 4. 基础变换（裁剪/旋转/颜色抖动）
                    img_tensor = self.img_transforms(img)
                    augmented_cams[cam_name].append(img_tensor)

            # 堆叠增强后的图像（T, C, H, W），然后转换为numpy
            head_cam = torch.stack(augmented_cams['head']).numpy()
            front_cam = torch.stack(augmented_cams['front']).numpy()
            left_cam = torch.stack(augmented_cams['left']).numpy()
            right_cam = torch.stack(augmented_cams['right']).numpy()
        else:
            # 不增强时，数据已经是(T, C, H, W)格式，只需归一化
            head_cam = head_cam.astype(np.float32) / 255.0
            front_cam = front_cam.astype(np.float32) / 255.0
            left_cam = left_cam.astype(np.float32) / 255.0
            right_cam = right_cam.astype(np.float32) / 255.0


        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W
                'front_cam': front_cam, # T, 3, H, W
                'left_cam': left_cam, # T, 3, H, W
                'right_cam': right_cam, # T, 3, H, W
                "agent_pos": agent_pos,  # T, D
            },
            "action": sample["action"].astype(np.float32),  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            # 1. 加载原始样本
            # 获取idx对应的原始样本
            sample = self.sampler.sample_sequence(idx)
            data = self._sample_to_data(sample)  # 转为观测和动作的字典格式
            # 应用Mix-Up 
            # 开启use_mixup并且满足随机数小于设定的概率条件
            if self.aug_config['use_mixup'] and random.random() < self.aug_config['mixup']['prob']:
                # 随机选择一个样本进行Mix-Up，并转换成相同的格式
                mix_idx = random.randint(0, len(self) - 1)
                mix_sample = self.sampler.sample_sequence(mix_idx)
                mix_data = self._sample_to_data(mix_sample)
                # 生成混合系数
                # 使用Beta分布生成混合系数，并确保λ ≥ 0.5（通过lam = max(lam, 1 - lam)实现
                lam = np.random.beta(
                    self.aug_config['mixup']['alpha'], 
                    self.aug_config['mixup']['alpha']
                )
                lam = max(lam, 1 - lam)

                # 混合所有相机图像、agent位置和动作
                data["obs"]["head_cam"] = lam * data["obs"]["head_cam"] + (1 - lam) * mix_data["obs"]["head_cam"]
                data["obs"]["front_cam"] = lam * data["obs"]["front_cam"] + (1 - lam) * mix_data["obs"]["front_cam"]
                data["obs"]["left_cam"] = lam * data["obs"]["left_cam"] + (1 - lam) * mix_data["obs"]["left_cam"]
                data["obs"]["right_cam"] = lam * data["obs"]["right_cam"] + (1 - lam) * mix_data["obs"]["right_cam"]
                data["obs"]["agent_pos"] = lam * data["obs"]["agent_pos"] + (1 - lam) * mix_data["obs"]["agent_pos"]
                data["action"] = lam * data["action"] + (1 - lam) * mix_data["action"]

            return dict_apply(data, torch.from_numpy)
            
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
        front_cam = samples['front_camera'].to(device, non_blocking=True) / 255.0
        left_cam = samples['left_camera'].to(device, non_blocking=True) / 255.0
        right_cam = samples['right_camera'].to(device, non_blocking=True) / 255.0
        action = samples["action"].to(device, non_blocking=True)
        return {
            "obs": {
                "head_cam": head_cam,  # B, T, 3, H, W
                'front_cam': front_cam, # B, T, 3, H, W
                'left_cam': left_cam, # B, T, 3, H, W
                'right_cam': right_cam, # B, T, 3, H, W
                "agent_pos": agent_pos,  # B, T, D
            },
            "action": action,  # B, T, D
        }


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
