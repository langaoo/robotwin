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
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt



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
        # 记录各相机的原始 (H, W)，用于增强后还原
        self.cam_hw = {
            'head_cam': tuple(self.replay_buffer['head_camera'].shape[-2:]),  # (H, W)
            'front_cam': tuple(self.replay_buffer['front_camera'].shape[-2:]),
            'left_cam': tuple(self.replay_buffer['left_camera'].shape[-2:]),
            'right_cam': tuple(self.replay_buffer['right_camera'].shape[-2:]),
        }

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

    def _save_augmentation_vis(self, original_data, augmented_data, idx, is_batch=False):
        """保存增强前后的图像对比"""
        save_dir = os.path.join("augmentation_vis", "batch" if is_batch else "single")
        os.makedirs(save_dir, exist_ok=True)
        
        t = 0
        for cam_name in ['head_cam', 'front_cam', 'left_cam', 'right_cam']:
            # 原始图像
            original_img = original_data["obs"][cam_name][t]
            original_img = np.transpose(original_img, (1, 2, 0)).astype(np.float32) / 255.0
            
            # 增强后图像
            augmented_img = augmented_data["obs"][cam_name][t]
            augmented_img = np.transpose(augmented_img, (1, 2, 0))
            
            # # ✅ 调试：打印数据范围
            # print(f"[调试] {cam_name} - 增强后范围: [{augmented_img.min():.3f}, {augmented_img.max():.3f}]")
            
            # 确保数据在 [0, 1] 范围
            if augmented_img.max() > 1.0:
                augmented_img = augmented_img / 255.0
            augmented_img = np.clip(augmented_img, 0, 1)  # ✅ 防御性裁剪
            
            # 绘制对比图
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(original_img)
            plt.title("Original")
            plt.axis("off")
            
            plt.subplot(122)
            plt.imshow(augmented_img)
            plt.title("Augmented")
            plt.axis("off")
            
            save_path = os.path.join(save_dir, f"idx_{idx}_cam_{cam_name}_frame_{t}.png")
            plt.savefig(save_path)
            plt.close()


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
        head_cam = sample["head_camera"].astype(np.float32)
        front_cam = sample['front_camera'].astype(np.float32)
        left_cam = sample['left_camera'].astype(np.float32)
        right_cam = sample['right_camera'].astype(np.float32)

     

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

    def _augment_single_camera(self, cam_data, target_hw):
        """增强单个相机的时序数据（T, C, H, W），并在增强后强制还原到 target_hw=(H, W)"""
        T = cam_data.shape[0]
        augmented_frames = []

        for t in range(T):
            # 转换为 (H, W, C) 格式（imgaug 要求）
            frame = np.transpose(cam_data[t], (1, 2, 0)).astype(np.uint8)

            # 1. 传感器噪声
            if self.sensor_noise_aug is not None:
                frame = self.sensor_noise_aug(image=frame)

            # 2. 相机视角抖动 + 透视变换
            if self.camera_jitter_aug is not None:
                frame = self.camera_jitter_aug(image=frame)

            # 3. 遮挡模拟
            if self.occlusion_aug is not None:
                frame = self.occlusion_aug(image=frame)

            # 基础变换到 tensor (C, H', W')，float32，[0,1]
            frame_tensor = self.img_transforms(frame)

            # ★ 关键：如果增强改变了分辨率，这里强制还原到模型登记的 (H, W)
            if frame_tensor.shape[1:] != target_hw:
                frame_tensor = TF.resize(frame_tensor, target_hw, antialias=True)
            

            augmented_frames.append(frame_tensor)

        return torch.stack(augmented_frames).numpy()

    def _augment_data(self, data):
            """统一增强逻辑，处理单样本数据"""
            obs = data["obs"]
            action = data["action"]
            need_aug = any([
                self.aug_config['use_random_crop'],
                self.aug_config['use_rotation'],
                self.aug_config['use_color_jitter'],
                self.aug_config['use_sensor_noise'],
                self.aug_config['use_camera_jitter'],
                self.aug_config['use_perspective'],
                self.aug_config['use_occlusion'],
            ])

            # 图像增强
            if need_aug:
                head_cam = self._augment_single_camera(obs["head_cam"], self.cam_hw['head_cam'])
                front_cam = self._augment_single_camera(obs["front_cam"], self.cam_hw['front_cam'])
                left_cam = self._augment_single_camera(obs["left_cam"], self.cam_hw['left_cam'])
                right_cam = self._augment_single_camera(obs["right_cam"], self.cam_hw['right_cam'])
            else:
                # 不增强时仅归一化
                head_cam = obs["head_cam"] / 255.0
                front_cam = obs["front_cam"] / 255.0
                left_cam = obs["left_cam"] / 255.0
                right_cam = obs["right_cam"] / 255.0

            # 构造增强后的观测数据
            augmented_obs = {
                "head_cam": head_cam,
                "front_cam": front_cam,
                "left_cam": left_cam,
                "right_cam": right_cam,
                "agent_pos": obs["agent_pos"].copy()
            }

            return {
            "obs": augmented_obs,
            "action": action
        }
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            # 1. 加载原始样本
            # 获取idx对应的原始样本
            sample = self.sampler.sample_sequence(idx)
            print(f"样本 {idx} 的原始序列索引范围: {self.sampler.indices[idx]}")  # 打印采样的原始帧范围
            data = self._sample_to_data(sample)  # 转为观测和动作的字典格式
            data = self._augment_data(data)  # 应用增强
            return dict_apply(data, torch.from_numpy)
            
        elif isinstance(idx, np.ndarray):
            # print(f"采样批量索引: {idx}")
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            # 逐样本增强并构建批量数据
            batch_obs = {
                "head_cam": [], "front_cam": [], "left_cam": [], 
                "right_cam": [], "agent_pos": []
            }
            batch_action = []
            
            for i in range(self.batch_size):
                # 提取单样本原始数据
                sample = {
                    "head_camera": self.buffers["head_camera"][i],
                    "front_camera": self.buffers["front_camera"][i],
                    "left_camera": self.buffers["left_camera"][i],
                    "right_camera": self.buffers["right_camera"][i],
                    "state": self.buffers["state"][i],
                    "action": self.buffers["action"][i],
                }
                # # 格式转换 + 增强
                # data = self._sample_to_data(sample)
                # data = self._augment_data(data)
                
                # 格式转换（未增强）
                original_data = self._sample_to_data(sample)
                # 应用增强
                data = self._augment_data(original_data)
                # # 可视化第1个批量中的第1个样本（避免过多文件）
                # if i == 0:
                #     self._save_augmentation_vis(original_data, data, idx[0], is_batch=True)


                # 收集到批量列表
                for k in batch_obs:
                    batch_obs[k].append(data["obs"][k])
                batch_action.append(data["action"])
            
            # 堆叠为批量张量（B, T, ...）
            batch_data = {
                "obs": {
                    k: np.stack(v, axis=0) for k, v in batch_obs.items()
                },
                "action": np.stack(batch_action, axis=0)
            }
            return dict_apply(batch_data, torch.from_numpy)
            # return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        
        return {
            "obs": {
                "head_cam": samples["obs"]["head_cam"].to(device, non_blocking=True),
                "front_cam": samples["obs"]["front_cam"].to(device, non_blocking=True),
                "left_cam": samples["obs"]["left_cam"].to(device, non_blocking=True),
                "right_cam": samples["obs"]["right_cam"].to(device, non_blocking=True),
                "agent_pos": samples["obs"]["agent_pos"].to(device, non_blocking=True),
            },
            "action": samples["action"].to(device, non_blocking=True),
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
