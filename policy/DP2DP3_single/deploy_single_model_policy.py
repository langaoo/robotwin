"""单模型推理部署 - 测试语义是否被融合破坏

与直接融合的区别:
- deploy_direct_fusion_policy.py: 4模型融合 → Head
- deploy_single_model_policy.py: 单模型 → Head (本文件)
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from collections import deque

# 添加路径
current_file_path = os.path.abspath(__file__)
policy_dir = os.path.dirname(current_file_path)
features_model_dir = os.path.join(policy_dir, "features_model")
sys.path.insert(0, features_model_dir)

from PIL import Image

# 导入特征提取器 (单模型)
from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors

# 导入正版DP
HAS_OFFICIAL_DP = False
try:
    DP_OUTER = Path(features_model_dir) / "third_party" / "DP" / "diffusion_policy"
    if DP_OUTER.exists():
        sys.path.insert(0, str(DP_OUTER))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        HAS_OFFICIAL_DP = True
        print("[INFO] 正版DP modules imported")
except ImportError as e:
    print(f"[WARNING] 正版DP导入失败: {e}")


class SingleModelDPPolicy(nn.Module):
    """单模型 + Diffusion Policy"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 2,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        
        # ✅ LinearNormalizer自动管理归一化
        self.normalizer = LinearNormalizer()
        
        # 观测编码器
        obs_encoder_dim = obs_dim * n_obs_steps
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Diffusion UNet
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=256,
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        
        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
    def forward(self, obs):
        """
        推理采样
        obs: [B, To, D]
        返回: [B, Ta, A] 原始尺度动作
        """
        B = obs.shape[0]
        device = obs.device
        
        # 编码观测
        obs_flat = obs.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        
        # 初始化噪声
        action = torch.randn((B, self.horizon, self.action_dim), device=device)
        
        # 去噪采样
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        
        # ✅ 自动反归一化
        action = self.normalizer['action'].unnormalize(action)
        
        return action


def encode_obs(observation):
    """编码观测"""
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    
    head_cam = observation["observation"]["head_camera"]["rgb"]
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0
    obs['head_cam'] = head_cam
    
    return obs


class SingleModelInference:
    """单模型推理包装器"""
    
    def __init__(self, usr_args):
        self.usr_args = usr_args
        self.gpu_id = 0
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.replan_every_call = True
        self.n_action_exec = int(usr_args.get('n_action_exec', 4))
        
        # 模型配置
        self.features_model_dir = Path(features_model_dir)
        self.policy_dir = Path(policy_dir)
        self.task_name = usr_args['task_name']
        self.ckpt_setting = usr_args.get('ckpt_setting', 'demo_clean')
        self.expert_data_num = usr_args.get('expert_data_num', 50)
        self.seed = usr_args.get('seed', 0)
        self.checkpoint_num = usr_args.get('checkpoint_num', 'best')
        self.model_name = usr_args.get('model_name', 'dinov3')  # 默认dinov3
        
        # Checkpoint路径
        ckpt_dir_name = f"{self.task_name}-{self.ckpt_setting}-{self.expert_data_num}-{self.model_name}"
        
        possible_dirs = [
            self.policy_dir / "checkpoints_single_model" / ckpt_dir_name,
            self.features_model_dir / "checkpoints_single_model" / ckpt_dir_name,
        ]
        
        ckpt_dir = None
        for d in possible_dirs:
            if d.exists():
                ckpt_dir = d
                break
        
        if ckpt_dir is None:
            raise FileNotFoundError(f"Checkpoint directory not found in: {possible_dirs}")
        
        if str(self.checkpoint_num).lower() == 'best':
            self.ckpt_path = ckpt_dir / "best.ckpt"
            if not self.ckpt_path.exists():
                ckpt_files = list(ckpt_dir.glob("*.ckpt"))
                if not ckpt_files:
                    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
                self.ckpt_path = max(ckpt_files, key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        else:
            self.ckpt_path = ckpt_dir / f"{int(self.checkpoint_num)}.ckpt"
            if not self.ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")
        
        print(f"[SingleModel] Loading checkpoint: {self.ckpt_path}")
        
        # 加载模型
        self._load_models()
        
        # 观测缓冲区
        self.obs_buffer = deque(maxlen=self.n_obs_steps)
        self.action_queue = deque()
        
        print(f"[SingleModel] Model initialization complete")
        print(f"  Task: {self.task_name}")
        print(f"  Model: {self.model_name}")
        print(f"  Horizon: {self.horizon}, N_obs_steps: {self.n_obs_steps}")
    
    def _load_models(self):
        """加载模型"""
        ckpt = torch.load(self.ckpt_path, map_location='cpu', weights_only=False)
        
        # 从checkpoint恢复配置
        config = ckpt.get('config', {})
        self.horizon = config.get('data', {}).get('horizon', 8)
        self.n_obs_steps = config.get('data', {}).get('n_obs_steps', 2)
        self.model_name = config.get('model', {}).get('name', 'dinov3')
        
        print(f"[SingleModel] Config: horizon={self.horizon}, n_obs_steps={self.n_obs_steps}")
        
        # 1. 加载特征提取器 (单模型)
        print(f"[SingleModel] Loading {self.model_name} backbone...")
        if torch.cuda.device_count() > 1:
            gpu_ids = [self.gpu_id, (self.gpu_id + 1) % torch.cuda.device_count()]
        else:
            gpu_ids = [self.gpu_id]
        
        self.feature_extractor = MultiGPUFeatureExtractors(gpu_ids=gpu_ids)
        
        # 获取模型索引
        model_idx_map = {'croco': 0, 'vggt': 1, 'dinov3': 2, 'da3': 3}
        self.model_idx = model_idx_map.get(self.model_name, 2)
        self.model_dim_map = {0: 1024, 1: 2048, 2: 768, 3: 2048}
        self.obs_dim = self.model_dim_map[self.model_idx]
        
        print(f"  Model: {self.model_name} (idx={self.model_idx}, dim={self.obs_dim})")
        
        # 2. 创建Policy
        print("[SingleModel] Creating Policy...")
        self.policy = SingleModelDPPolicy(
            obs_dim=self.obs_dim,
            action_dim=14,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            num_inference_steps=config.get('policy', {}).get('num_inference_steps', 100),
        )
        
        # 3. 加载权重
        print("[SingleModel] Loading weights...")
        policy_state = ckpt.get('policy', {})
        
        if any(k.startswith('module.') for k in policy_state.keys()):
            policy_state = {k.replace('module.', ''): v for k, v in policy_state.items()}
        
        missing, unexpected = self.policy.load_state_dict(policy_state, strict=False)
        if missing:
            print(f"[WARNING] Missing keys: {missing}")
        if unexpected:
            print(f"[WARNING] Unexpected keys: {unexpected}")
        
        # 4. ✅ 加载normalizer
        if 'normalizer' in ckpt:
            print("[SingleModel] Loading normalizer...")
            self.policy.normalizer.load_state_dict(ckpt['normalizer'])
            print(f"  Action norm: min={self.policy.normalizer['action'].params_dict['input_stats']['min'][:3]}...")
            print(f"  Action norm: max={self.policy.normalizer['action'].params_dict['input_stats']['max'][:3]}...")
        else:
            print("[WARNING] Normalizer not found in checkpoint!")
        
        self.policy = self.policy.to(self.device).eval()
        
        print("[SingleModel] All models loaded successfully!")
    
    def reset(self):
        """重置状态"""
        self.obs_buffer.clear()
        self.action_queue.clear()
    
    def get_action(self, obs):
        """获取动作"""
        self.obs_buffer.append(obs)
        
        while len(self.obs_buffer) < self.n_obs_steps:
            self.obs_buffer.append(self.obs_buffer[0])
        
        if self.replan_every_call or len(self.action_queue) == 0:
            # 1. 准备图像
            images = []
            for o in self.obs_buffer:
                img_np = (o['head_cam'].transpose(1, 2, 0) * 255.0).astype(np.uint8)
                img_np = np.clip(img_np, 0, 255)
                img_pil = Image.fromarray(img_np, mode='RGB')
                images.append(img_pil)
            
            # 2. 提取单模型特征
            features_np = self.feature_extractor.extract_batch(images)  # [To, 4, 2048]
            
            # 提取当前模型的特征
            model_feat = features_np[:, self.model_idx, :self.obs_dim]  # [To, D]
            
            # 转tensor: [1, To, D]
            features = torch.from_numpy(model_feat).float().unsqueeze(0).to(self.device)
            
            # 3. 通过policy预测
            with torch.no_grad():
                action_pred = self.policy(features)  # [1, Ta, A] - 已自动反归一化
            
            action_pred = action_pred.squeeze(0).cpu().numpy()  # [Ta, A]
            
            print(f"[DEBUG] 模型输出 ({self.model_name}):")
            print(f"  Range: [{action_pred.min():.3f}, {action_pred.max():.3f}]")
            print(f"  Mean: {action_pred.mean():.3f}, Std: {action_pred.std():.3f}")
            
            # Clip安全限制
            action_pred = np.clip(action_pred, -3.0, 3.0)
            
            # 加入队列
            self.action_queue.clear()
            self.action_queue.extend(action_pred)
        
        n_action_exec = min(self.n_action_exec, len(self.action_queue))
        return list(self.action_queue)[:n_action_exec]
    
    def update_obs(self, obs):
        """更新观测"""
        self.obs_buffer.append(obs)
    
    def pop_action(self):
        """弹出动作"""
        if len(self.action_queue) > 0:
            return self.action_queue.popleft()
        return np.zeros(14, dtype=np.float32)


def get_model(usr_args):
    """RoBoTwin标准接口"""
    model = SingleModelInference(usr_args)
    return model


def eval(TASK_ENV, model, observation):
    """RoBoTwin标准接口"""
    obs = encode_obs(observation)
    
    actions = model.get_action(obs)
    
    for action in actions:
        TASK_ENV.take_action(action)
        model.pop_action()
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    """RoBoTwin标准接口"""
    model.reset()
