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
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision.transforms import v2 as T
import time
from kornia.augmentation import (
    ColorJiggle, RandomResizedCrop, RandomRotation,
    RandomGaussianNoise, RandomGaussianBlur,
    RandomAffine, RandomPerspective, RandomErasing
)
from einops import rearrange

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
        # self.img_transforms = self._build_img_transforms()
        # self.sensor_noise_aug = self._build_sensor_noise_aug()
        # self.camera_jitter_aug = self._build_camera_jitter_aug()
        # self.occlusion_aug = self._build_occlusion_aug()
        # 记录各相机的原始 (H, W)，用于增强后还原
        # self.cam_hw = {
        #     'head_cam': tuple(self.replay_buffer['head_camera'].shape[-2:]),  # (H, W)
        #     'front_cam': tuple(self.replay_buffer['front_camera'].shape[-2:]),
        #     'left_cam': tuple(self.replay_buffer['left_camera'].shape[-2:]),
        #     'right_cam': tuple(self.replay_buffer['right_camera'].shape[-2:]),
        # }

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
            # 注意: random_crop会保持原始分辨率,只是随机裁剪和缩放后resize回原尺寸
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

    def _save_augmentation_vis(self, original_cams, augmented_cams, batch_idx, save_dir="augmentation_vis"):
        """保存增强前后的图像对比
        Args:
            original_cams: 原始图像张量 (4, B, L, C, H, W)
            augmented_cams: 增强后图像张量 (4, B, L, C, H, W)
            batch_idx: 当前batch的索引
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        cam_names = ['head_cam', 'front_cam', 'left_cam', 'right_cam']
        
        # 只保存第一个样本的第一帧
        sample_idx = 0
        time_idx = 0
        
        for cam_idx, cam_name in enumerate(cam_names):
            # 原始图像 (C, H, W) - 转为 (H, W, C)
            original_img = original_cams[cam_idx, sample_idx, time_idx].cpu().numpy()
            original_img = np.transpose(original_img, (1, 2, 0))
            
            # 增强后图像 (C, H, W) - 转为 (H, W, C)
            augmented_img = augmented_cams[cam_idx, sample_idx, time_idx].cpu().numpy()
            augmented_img = np.transpose(augmented_img, (1, 2, 0))
            
            # 确保数据在 [0, 1] 范围
            original_img = np.clip(original_img / 255.0 if original_img.max() > 1.0 else original_img, 0, 1)
            augmented_img = np.clip(augmented_img / 255.0 if augmented_img.max() > 1.0 else augmented_img, 0, 1)
            
            # 绘制对比图
            plt.figure(figsize=(12, 5))
            
            plt.subplot(121)
            plt.imshow(original_img)
            plt.title(f"Original - {cam_name}", fontsize=14)
            plt.axis("off")
            
            plt.subplot(122)
            plt.imshow(augmented_img)
            plt.title(f"Augmented - {cam_name}", fontsize=14)
            plt.axis("off")
            
            # 保存文件
            save_path = os.path.join(save_dir, f"batch_{batch_idx:04d}_{cam_name}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
        print(f"✅ 增强可视化已保存到: {save_dir}/batch_{batch_idx:04d}_*.png")


    def _build_img_transforms(self):
        """构建支持批量处理的图像增强管道"""
        transforms_list = [T.ToImage()]  # 接受 (N, C, H, W) 或 (N, H, W, C)

        # 随机裁剪
        if self.aug_config['use_random_crop']:
            crop_size = tuple(self.aug_config['crop_size'])
            transforms_list.append(T.RandomResizedCrop(
                size=crop_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                antialias=True
            ))

        # 随机旋转
        if self.aug_config['use_rotation']:
            degrees = self.aug_config['rotation_degrees']
            transforms_list.append(T.RandomRotation(degrees=degrees))

        # 颜色抖动
        if self.aug_config['use_color_jitter']:
            jitter_cfg = self.aug_config['color_jitter']
            transforms_list.append(T.ColorJitter(
                brightness=jitter_cfg['brightness'],
                contrast=jitter_cfg['contrast'],
                saturation=jitter_cfg['saturation'],
                hue=jitter_cfg['hue']
            ))

        # 统一转为 float32 并缩放到 [0, 1]
        transforms_list.append(T.ToDtype(torch.float32, scale=True))
        return T.Compose(transforms_list)

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
        # val_set.img_transforms = val_set._build_img_transforms()
        # val_set.sensor_noise_aug = val_set._build_sensor_noise_aug()
        # val_set.camera_jitter_aug = val_set._build_camera_jitter_aug()
        # val_set.occlusion_aug = val_set._build_occlusion_aug()
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

    # def _augment_single_camera(self, cam_data, target_hw):
    #     """增强单个相机的时序数据（T, C, H, W），并在增强后强制还原到 target_hw=(H, W)"""
    #     seq_len = cam_data.shape[0]
    #     start_time = time.time()  # 记录开始时间
    #     # print("T=" + str(seq_len))

    #     # 1. 转换为 (T, H, W, C) 格式，满足 imgaug 的批量接口
    #     frames_hwcn = np.transpose(cam_data, (0, 2, 3, 1)).astype(np.uint8)

    #     # 2. 向量化执行各类图像增强（为线程安全，使用 deterministic 副本）
    #     if self.sensor_noise_aug is not None:
    #         frames_hwcn = self.sensor_noise_aug.to_deterministic()(images=frames_hwcn)
    #     if self.camera_jitter_aug is not None:
    #         frames_hwcn = self.camera_jitter_aug.to_deterministic()(images=frames_hwcn)
    #     if self.occlusion_aug is not None:
    #         frames_hwcn = self.occlusion_aug.to_deterministic()(images=frames_hwcn)

    #     # imgaug 可能返回 list，需要显式转回 ndarray
    #     frames_hwcn = np.asarray(frames_hwcn, dtype=np.uint8)

    #     # 3. 转回 (T, C, H, W) 并一次性送入 torchvision 的批量变换
    #     frames_tensor = torch.from_numpy(frames_hwcn).permute(0, 3, 1, 2)
    #     frames_tensor = self.img_transforms(frames_tensor)

    #     # 4. 如果增强改变了分辨率，则统一插值回目标尺寸
    #     if frames_tensor.shape[-2:] != target_hw:
    #         frames_tensor = TF.resize(frames_tensor, list(target_hw), antialias=True)

    #     # print("增强总用时-----：" + str(time.time() - start_time))
    #     return frames_tensor.contiguous().numpy()

    # def _augment_batch_camera(self, cam_data_batch, target_hw):
    #     """批量增强多个样本的单个相机数据（B, T, C, H, W）"""
    #     batch_size, seq_len = cam_data_batch.shape[:2]
        
    #     # 1. 合并batch和time维度 (B*T, C, H, W)
    #     cam_data_flat = cam_data_batch.reshape(-1, *cam_data_batch.shape[2:])
        
    #     # 2. 转换为 (B*T, H, W, C) 格式，满足 imgaug 的批量接口
    #     frames_hwcn = np.transpose(cam_data_flat, (0, 2, 3, 1)).astype(np.uint8)
        
    #     # 3. 向量化执行各类图像增强（为线程安全，使用 deterministic 副本）
    #     if self.sensor_noise_aug is not None:
    #         frames_hwcn = self.sensor_noise_aug.to_deterministic()(images=frames_hwcn)
    #     if self.camera_jitter_aug is not None:
    #         frames_hwcn = self.camera_jitter_aug.to_deterministic()(images=frames_hwcn)
    #     if self.occlusion_aug is not None:
    #         frames_hwcn = self.occlusion_aug.to_deterministic()(images=frames_hwcn)
        
    #     # imgaug 可能返回 list，需要显式转回 ndarray
    #     frames_hwcn = np.asarray(frames_hwcn, dtype=np.uint8)
        
    #     # 4. 转回 (B*T, C, H, W) 并一次性送入 torchvision 的批量变换
    #     frames_tensor = torch.from_numpy(frames_hwcn).permute(0, 3, 1, 2)
    #     frames_tensor = self.img_transforms(frames_tensor)
        
    #     # 5. 如果增强改变了分辨率，则统一插值回目标尺寸
    #     if frames_tensor.shape[-2:] != target_hw:
    #         frames_tensor = TF.resize(frames_tensor, list(target_hw), antialias=True)
        
    #     # 6. 恢复batch和time维度 (B, T, C, H, W)
    #     result = frames_tensor.contiguous().numpy()
    #     result = result.reshape(batch_size, seq_len, *result.shape[1:])
        
    #     return result

    # def _augment_data(self, data):
    #         """统一增强逻辑，处理单样本数据"""
    #         obs = data["obs"]
    #         action = data["action"]
    #         need_aug = any([
    #             self.aug_config['use_random_crop'],
    #             self.aug_config['use_rotation'],
    #             self.aug_config['use_color_jitter'],
    #             self.aug_config['use_sensor_noise'],
    #             self.aug_config['use_camera_jitter'],
    #             self.aug_config['use_perspective'],
    #             self.aug_config['use_occlusion'],
    #         ])

    #         # 图像增强
    #         if need_aug:
    #             tasks = {
    #                 'head_cam': (obs['head_cam'], self.cam_hw['head_cam']),
    #                 'front_cam': (obs['front_cam'], self.cam_hw['front_cam']),
    #                 'left_cam': (obs['left_cam'], self.cam_hw['left_cam']),
    #                 'right_cam': (obs['right_cam'], self.cam_hw['right_cam'])
    #             }
    #             results = {}
    #             try:
    #                 with ThreadPoolExecutor(max_workers=4) as ex:
    #                     future_map = {name: ex.submit(self._augment_single_camera, data, hw)
    #                                   for name, (data, hw) in tasks.items()}
    #                     for name, fut in future_map.items():
    #                         results[name] = fut.result()
    #             except Exception as e:
    #                 # 若并行失败，回退到串行，保证训练不中断
    #                 print(f"[warn] 并行增强失败，回退串行：{e}")
    #                 for name, (data_arr, hw) in tasks.items():
    #                     results[name] = self._augment_single_camera(data_arr, hw)

    #             head_cam = results['head_cam']
    #             front_cam = results['front_cam']
    #             left_cam = results['left_cam']
    #             right_cam = results['right_cam']            
    #         else:
    #             # 不增强时仅归一化
    #             head_cam = obs["head_cam"] / 255.0
    #             front_cam = obs["front_cam"] / 255.0
    #             left_cam = obs["left_cam"] / 255.0
    #             right_cam = obs["right_cam"] / 255.0

    #         # 构造增强后的观测数据
    #         augmented_obs = {
    #             "head_cam": head_cam,
    #             "front_cam": front_cam,
    #             "left_cam": left_cam,
    #             "right_cam": right_cam,
    #             "agent_pos": obs["agent_pos"].copy()
    #         }

    #         return {
    #         "obs": augmented_obs,
    #         "action": action
    #     }

    # def _augment_batch_data(self, batch_data):
    #     """批量增强逻辑，处理整个batch的数据 (B, T, ...)"""
    #     obs_batch = batch_data["obs"]
    #     action_batch = batch_data["action"]
        
    #     need_aug = any([
    #         self.aug_config['use_random_crop'],
    #         self.aug_config['use_rotation'],
    #         self.aug_config['use_color_jitter'],
    #         self.aug_config['use_sensor_noise'],
    #         self.aug_config['use_camera_jitter'],
    #         self.aug_config['use_perspective'],
    #         self.aug_config['use_occlusion'],
    #     ])

    #     # 图像批量增强
    #     if need_aug:
    #         tasks = {
    #             'head_cam': (obs_batch['head_cam'], self.cam_hw['head_cam']),
    #             'front_cam': (obs_batch['front_cam'], self.cam_hw['front_cam']),
    #             'left_cam': (obs_batch['left_cam'], self.cam_hw['left_cam']),
    #             'right_cam': (obs_batch['right_cam'], self.cam_hw['right_cam'])
    #         }
    #         results = {}
    #         try:
    #             # 并行处理4个相机
    #             with ThreadPoolExecutor(max_workers=4) as ex:
    #                 future_map = {name: ex.submit(self._augment_batch_camera, data, hw)
    #                               for name, (data, hw) in tasks.items()}
    #                 for name, fut in future_map.items():
    #                     results[name] = fut.result()
    #         except Exception as e:
    #             # 若并行失败，回退到串行，保证训练不中断
    #             # print(f"[warn] 并行批量增强失败，回退串行：{e}")
    #             for name, (data_arr, hw) in tasks.items():
    #                 results[name] = self._augment_batch_camera(data_arr, hw)

    #         head_cam = results['head_cam']
    #         front_cam = results['front_cam']
    #         left_cam = results['left_cam']
    #         right_cam = results['right_cam']
    #     else:
    #         # 不增强时仅归一化（向量化操作）
    #         head_cam = obs_batch["head_cam"] / 255.0
    #         front_cam = obs_batch["front_cam"] / 255.0
    #         left_cam = obs_batch["left_cam"] / 255.0
    #         right_cam = obs_batch["right_cam"] / 255.0

    #     # 构造增强后的批量观测数据
    #     augmented_obs = {
    #         "head_cam": head_cam,
    #         "front_cam": front_cam,
    #         "left_cam": left_cam,
    #         "right_cam": right_cam,
    #         "agent_pos": obs_batch["agent_pos"].copy()
    #     }

    #     return {
    #         "obs": augmented_obs,
    #         "action": action_batch
    #     }
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            # 1. 加载原始样本
            # 获取idx对应的原始样本
            sample = self.sampler.sample_sequence(idx)
            # print(f"样本 {idx} 的原始序列索引范围: {self.sampler.indices[idx]}")  # 打印采样的原始帧范围
            data = self._sample_to_data(sample)  # 转为观测和动作的字典格式
            # data = self._augment_data(data)  # 应用增强
            return dict_apply(data, torch.from_numpy)
            
        elif isinstance(idx, np.ndarray):
            # print(f"采样批量索引: {idx}")
            assert len(idx) == self.batch_size
            
            # 批量采样所有相机和状态数据
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            
            # 直接构建批量数据字典（避免逐样本循环）
            batch_data = {
                "obs": {
                    "head_cam": self.buffers["head_camera"].astype(np.float32),  # (B, T, C, H, W)
                    "front_cam": self.buffers["front_camera"].astype(np.float32),
                    "left_cam": self.buffers["left_camera"].astype(np.float32),
                    "right_cam": self.buffers["right_camera"].astype(np.float32),
                    "agent_pos": self.buffers["state"].astype(np.float32),  # (B, T, D)
                },
                "action": self.buffers["action"].astype(np.float32)  # (B, T, D)
            }
            
            return dict_apply(batch_data, torch.from_numpy)
            # return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        batch_size = samples["obs"]["head_cam"].shape[0]
        
        # print("head_cam shape before aug:", samples["obs"]["head_cam"].shape)
        # start_time = time.time()  # 记录开始时间
        
        # 将所有摄像头数据堆叠成一个大张量 (4, B, L, C, H, W)
        all_cams = torch.stack([
            samples["obs"]["head_cam"],
            samples["obs"]["front_cam"],
            samples["obs"]["left_cam"],
            samples["obs"]["right_cam"]
        ], dim=0)
        
        # 保存原始图像用于对比
        # all_cams_original = all_cams.clone()
        
        # 重排为 (4*B*L, C, H, W) - 一次性处理所有摄像头的所有帧
        all_cams_flat = rearrange(all_cams, 'n b l c h w -> (n b l) c h w')
        
        # ✅ 关键修复1: 先归一化到 [0, 1] 范围 (Kornia期望的输入范围)
        all_cams_flat = all_cams_flat.float() / 255.0
        
        # 移到GPU上再做增强（Kornia在GPU上速度快很多）
        all_cams_flat = all_cams_flat.to(device, non_blocking=True)
        
        # 构建增强列表
        aug_list = []
        
        # 1. 颜色抖动
        if self.aug_config['use_color_jitter']:
            jitter_cfg = self.aug_config['color_jitter']
            aug_list.append(ColorJiggle(
                brightness=jitter_cfg['brightness'],
                contrast=jitter_cfg['contrast'],
                saturation=jitter_cfg['saturation'],
                hue=jitter_cfg['hue'],
                p=jitter_cfg['prob']
            ))
        
        # 2. 随机裁剪和缩放 (保持原始分辨率)
        if self.aug_config['use_random_crop']:
            # 使用原始图像的尺寸,不改变分辨率
            crop_cfg = self.aug_config['random_crop']
            aug_list.append(RandomResizedCrop(
                (all_cams_flat.shape[-2], all_cams_flat.shape[-1]),  # 保持原始H和W
                scale=crop_cfg['scale'], 
                ratio=crop_cfg['ratio'], 
                p=crop_cfg['prob']
            ))
        
        # 3. 随机旋转
        if self.aug_config['use_rotation']:
            degrees_cfg = self.aug_config['rotation_degrees']
            aug_list.append(RandomRotation(
                degrees=degrees_cfg['degrees'], 
                p=degrees_cfg['prob']
            ))
        
        # 4. 传感器噪声 (高斯噪声 + 高斯模糊)
        if self.aug_config['use_sensor_noise']:
            noise_cfg = self.aug_config['sensor_noise']
            # 高斯噪声
            aug_list.append(RandomGaussianNoise(
                mean=noise_cfg['mean'],
                std=noise_cfg['gaussian_scale'],
                p=noise_cfg['prob']
            ))
            # 高斯模糊
            blur_sigma = noise_cfg['blur_sigma']
            if blur_sigma[1] > 0:  # 确保有效的sigma范围
                aug_list.append(RandomGaussianBlur(
                    kernel_size=(3, 3),
                    sigma=blur_sigma,
                    p=noise_cfg['prob']
                ))
        
        # 5. 相机视角抖动 (仿射变换)
        if self.aug_config['use_camera_jitter']:
            jitter_cfg = self.aug_config['camera_jitter']
            aug_list.append(RandomAffine(
                degrees=0,  # 旋转已经单独处理
                translate=(jitter_cfg['translate_percent'], jitter_cfg['translate_percent']),
                scale=jitter_cfg['scale'],
                shear=jitter_cfg['shear'],
                p=jitter_cfg['prob']
            ))
        
        # 6. 透视变换
        if self.aug_config['use_perspective']:
            persp_cfg = self.aug_config['perspective']
            aug_list.append(RandomPerspective(
                distortion_scale=persp_cfg['scale'][1],  # 使用最大值
                p=persp_cfg['prob']
            ))
        
        # 7. 遮挡模拟 (Random Erasing)
        if self.aug_config['use_occlusion']:
            occ_cfg = self.aug_config['occlusion']
            aug_list.append(RandomErasing(
                scale=occ_cfg['scale'],
                ratio=occ_cfg['ratio'],
                value=0.0,  # 黑色遮挡
                p=occ_cfg['prob']
            ))
        
        # 应用所有增强（GPU并行执行）
        all_cams_aug = all_cams_flat
        for aug in aug_list:
            all_cams_aug = aug(all_cams_aug)
        
        # ✅ 关键修复7: 确保输出在合法范围内
        all_cams_aug = torch.clamp(all_cams_aug, 0.0, 1.0)
        
        # ✅ 关键修复8: 转回 [0, 255] 范围
        all_cams_aug = (all_cams_aug * 255.0).to(torch.uint8).float()
        
        # 重新分离各个摄像头 (4*B*L, C, H, W) -> (4, B, L, C, H, W)
        all_cams_aug = rearrange(all_cams_aug, '(n b l) c h w -> n b l c h w', 
                                 n=4, b=batch_size)
        
        
        
        # # 保存增强可视化（只在前几个batch保存，避免过多文件）
        # if not hasattr(self, '_vis_counter'):
        #     self._vis_counter = 0
        # if self._vis_counter < 5:  # 只保存前5个batch
        #     self._save_augmentation_vis(all_cams_original, all_cams_aug, self._vis_counter)
        #     self._vis_counter += 1

        # print("增强总用时-----：" + str(time.time() - start_time))
        return {
            "obs": {
                "head_cam": all_cams_aug[0].to(device, non_blocking=True),
                "front_cam": all_cams_aug[1].to(device, non_blocking=True),
                "left_cam": all_cams_aug[2].to(device, non_blocking=True),
                "right_cam": all_cams_aug[3].to(device, non_blocking=True),
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
