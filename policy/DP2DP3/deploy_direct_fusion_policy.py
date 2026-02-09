"""DP2DP3 Direct Fusion Policy Deployment for RoBoTwin

这个文件是专门用于**直接融合训练**模型的推理入口。
与 deploy_policy.py 的区别：
- deploy_policy.py: RGB → 4模型 → 对齐编码器(RGB2PC) → Head
- deploy_direct_fusion_policy.py: RGB → 4模型 → 简单融合 → Head (无对齐模块)

架构：
1. RGB 图像 → 4 个视觉模型 (CroCo/VGGT/DINOv3/DA3) → RGB 特征
2. RGB 特征 → 简单融合器 (weighted/concat) → 融合特征
3. 融合特征 → Diffusion Policy Head → 动作输出
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from collections import deque

# 添加 features_model 到路径
current_file_path = os.path.abspath(__file__)
policy_dir = os.path.dirname(current_file_path)
features_model_dir = os.path.join(policy_dir, "features_model")
sys.path.insert(0, features_model_dir)

from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors
from PIL import Image

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


class SimpleFusionEncoder(nn.Module):
    """简单融合编码器: 4模型特征 → 融合 → 输出"""
    
    def __init__(
        self,
        in_dims=(1024, 2048, 768, 2048),
        fusion_type='weighted',
        out_dim=1280,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.fusion_type = fusion_type
        self.out_dim = out_dim
        
        if fusion_type == 'weighted':
            # 可学习权重
            self.fusion_weights = nn.Parameter(torch.ones(len(in_dims)) / len(in_dims))
            # 投影层：将每个模型的特征投影到统一维度
            # 注意：训练代码中使用的是 projectors，所以这里也要匹配
            self.projectors = nn.ModuleList([
                nn.Linear(in_dim, out_dim) for in_dim in in_dims
            ])
        elif fusion_type == 'concat':
            # 拼接后投影
            total_dim = sum(in_dims)
            # ✅ 与训练端保持一致（两层 MLP + LayerNorm，参数名为 projector）
            self.projector = nn.Sequential(
                nn.Linear(total_dim, out_dim * 2),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim),
                nn.LayerNorm(out_dim),
            )
        elif fusion_type == 'mean':
            # 简单平均，需要投影到相同维度
            self.projectors = nn.ModuleList([
                nn.Linear(in_dim, out_dim) for in_dim in in_dims
            ])
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, x):
        """
        x: [B, To, M, C_max] - 每个模型特征pad到最大维度
        返回: [B, To, D] 或 [B*To, D]
        
        注意：每个模型只使用其实际维度，忽略padding部分
        """
        if x.ndim == 4:
            B, To, M, _ = x.shape
            flat = True
            x = x.reshape(B * To, M, -1)
        else:
            flat = False
            B, M, _ = x.shape
            To = 1
        
        if self.fusion_type == 'weighted':
            # 加权融合
            # 可选: 推理时对融合权重做mask（用于排查某个模型破坏融合）
            if hasattr(self, "fusion_mask") and self.fusion_mask is not None:
                mask = self.fusion_mask.to(self.fusion_weights.device, dtype=self.fusion_weights.dtype)
                mask = torch.clamp(mask, min=0.0, max=1.0)
                # mask=0 的位置用极小值抑制
                masked_logits = self.fusion_weights + torch.log(mask + 1e-6)
                weights = torch.softmax(masked_logits, dim=0)
            else:
                weights = torch.softmax(self.fusion_weights, dim=0)
            fused = []
            for i in range(M):
                # 提取每个模型的实际维度特征（去掉padding）
                feat = x[:, i, :self.in_dims[i]]  # [B*To, in_dim_i]
                # 投影到统一维度
                feat = self.projectors[i](feat)  # [B*To, D]
                fused.append(feat * weights[i])
            output = torch.stack(fused, dim=0).sum(dim=0)  # [B*To, D]
        
        elif self.fusion_type == 'concat':
            # 拼接融合
            features = []
            for i in range(M):
                feat = x[:, i, :self.in_dims[i]]  # [B*To, in_dim_i]
                features.append(feat)
            concat_feat = torch.cat(features, dim=-1)  # [B*To, sum(in_dims)]
            output = self.projector(concat_feat)  # [B*To, D]
        
        elif self.fusion_type == 'mean':
            # 平均融合
            projected = []
            for i in range(M):
                feat = x[:, i, :self.in_dims[i]]  # [B*To, in_dim_i]
                feat = self.projectors[i](feat)  # [B*To, D]
                projected.append(feat)
            output = torch.stack(projected, dim=0).mean(dim=0)  # [B*To, D]
        
        if flat:
            output = output.reshape(B, To, -1)
        
        return output


class DirectFusionDPPolicy(nn.Module):
    """直接融合 + Diffusion Policy"""
    
    def __init__(
        self,
        fusion_encoder: SimpleFusionEncoder,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 3,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版DP未加载，无法使用DirectFusionDPPolicy")
        
        self.fusion_encoder = fusion_encoder
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.use_proprio = False
        # ✅ 使用LinearNormalizer管理归一化
        self.normalizer = LinearNormalizer()
        self.use_normalizer = False
        
        # 观测维度 = 融合后特征维度 * 观测步数
        obs_dim = fusion_encoder.out_dim * n_obs_steps
        
        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
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
    
    def forward(self, rgb_feats, agent_pos=None):
        """
        推理模式
        rgb_feats: [B, To, M, C] - 4个模型的RGB特征
        返回: [B, Ta, A] - 动作序列
        """
        B = rgb_feats.shape[0]
        device = rgb_feats.device
        
        # 1. 融合特征
        fused_feats = self.fusion_encoder(rgb_feats)  # [B, To, D]
        
        # 2. 编码观测
        obs_flat = fused_feats.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        
        # 3. 初始化随机噪声
        action = torch.randn((B, self.horizon, self.action_dim), device=device)
        
        # 4. 设置推理步数
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        # 5. 逐步去噪
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        
        # ✅ 反归一化到动作原始尺度（若checkpoint提供normalizer）
        if self.use_normalizer:
            action = self.normalizer.unnormalize({'action': action})['action']
        return action


class DirectFusionProprioPolicy(nn.Module):
    """直接融合 + proprio + Diffusion Policy（DP-aligned: agent_pos 直接拼接）"""
    
    def __init__(
        self,
        fusion_encoder: SimpleFusionEncoder,
        proprio_dim: int = 14,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 3,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版DP未加载")
        
        self.fusion_encoder = fusion_encoder
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.use_proprio = True
        self.normalizer = LinearNormalizer()
        self.use_normalizer = True
        
        per_step_dim = fusion_encoder.out_dim + proprio_dim
        obs_encoder_dim = per_step_dim * n_obs_steps
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=256,
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
    def forward(self, rgb_feats, agent_pos=None):
        """
        推理
        rgb_feats: [B, To, M, C]
        agent_pos: [B, To, proprio_dim]
        """
        B = rgb_feats.shape[0]
        device = rgb_feats.device
        
        fused = self.fusion_encoder(rgb_feats)  # [B, To, D]
        
        if agent_pos is not None:
            nagent_pos = self.normalizer['agent_pos'].normalize(agent_pos)
            obs_combined = torch.cat([fused, nagent_pos], dim=-1)
        else:
            zeros = torch.zeros(B, fused.shape[1], self.proprio_dim, device=device)
            obs_combined = torch.cat([fused, zeros], dim=-1)
        
        obs_flat = obs_combined.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        
        action = torch.randn((B, self.horizon, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action, t.unsqueeze(0).expand(B).to(device), global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        
        action = self.normalizer.unnormalize({'action': action})['action']
        return action


def encode_obs(observation):
    """编码观测（兼容RoBoTwin格式）"""
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    
    head_cam = observation["observation"]["head_camera"]["rgb"]
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0
    obs['head_cam'] = head_cam
    
    return obs


class DirectFusionModel:
    """直接融合模型包装器"""
    
    def __init__(self, usr_args):
        self.usr_args = usr_args
        # GPU ID: 在 CUDA_VISIBLE_DEVICES 环境下始终使用 0（已经被映射）
        self.gpu_id = 0
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.replan_every_call = True
        self.n_action_exec = int(usr_args.get('n_action_exec', 4))
        
        # 模型路径
        self.features_model_dir = Path(features_model_dir)
        self.policy_dir = Path(policy_dir)
        self.task_name = usr_args['task_name']
        self.ckpt_setting = usr_args.get('ckpt_setting', 'demo_clean')
        self.expert_data_num = usr_args.get('expert_data_num', 50)
        self.seed = usr_args.get('seed', 0)
        self.checkpoint_num = usr_args.get('checkpoint_num', 'best')
        
        # 构建checkpoint路径 - 直接融合的checkpoints在特殊目录
        ckpt_dir_name = f"{self.task_name}-{self.ckpt_setting}-{self.expert_data_num}-{self.seed}"
        ckpt_dir_name_proprio = f"{ckpt_dir_name}-proprio"

        # 允许通过环境变量指定checkpoint根目录（用于ws1等变体）
        # 例：DP2DP3_DIRECT_FUSION_CKPT_ROOTS=/path/to/checkpoints_direct_fusion_ws1
        ckpt_roots_env = os.environ.get("DP2DP3_DIRECT_FUSION_CKPT_ROOTS", "").strip()
        if ckpt_roots_env:
            ckpt_roots = [Path(p) for p in ckpt_roots_env.split(":") if p]
        else:
            ckpt_roots = [
                self.policy_dir / "checkpoints_direct_fusion",
                self.features_model_dir / "checkpoints_direct_fusion",
                self.policy_dir / "checkpoints_direct_fusion_ws1",
                self.features_model_dir / "checkpoints_direct_fusion_ws1",
            ]

        # proprio版优先搜索
        possible_dirs = [root / ckpt_dir_name_proprio for root in ckpt_roots] + \
                        [root / ckpt_dir_name for root in ckpt_roots]
        
        ckpt_dir = None
        for d in possible_dirs:
            if d.exists():
                ckpt_dir = d
                break
        
        if ckpt_dir is None:
            raise FileNotFoundError(f"Checkpoint directory not found in: {possible_dirs}")
        
        if str(self.checkpoint_num).lower() == 'best':
            ckpt_files = list(ckpt_dir.glob("*.ckpt"))
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
            # 找到best.ckpt或最新的
            best_ckpt = ckpt_dir / "best.ckpt"
            if best_ckpt.exists():
                self.head_ckpt_path = best_ckpt
            else:
                self.head_ckpt_path = max(ckpt_files, key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        else:
            self.head_ckpt_path = ckpt_dir / f"{int(self.checkpoint_num)}.ckpt"
            if not self.head_ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.head_ckpt_path}")
        
        print(f"[DirectFusion] Loading checkpoint: {self.head_ckpt_path}")
        
        # 加载模型
        self._load_models()
        
        # 观测缓冲区
        self.obs_buffer = deque(maxlen=self.n_obs_steps)
        self.action_queue = deque()
        
        print(f"[DirectFusion] Model initialization complete")
        print(f"  Task: {self.task_name}")
        print(f"  Horizon: {self.horizon}, N_obs_steps: {self.n_obs_steps}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Fusion type: {self.fusion_type}")
    
    def _load_models(self):
        """加载模型"""
        ckpt = torch.load(self.head_ckpt_path, map_location='cpu', weights_only=False)
        
        # 从checkpoint恢复配置
        config = ckpt.get('config', {})
        self.horizon = config.get('data', {}).get('horizon', 8)
        self.n_obs_steps = config.get('data', {}).get('n_obs_steps', 3)
        self.fusion_type = config.get('fusion', {}).get('type', 'weighted')
        self.fuse_dim = config.get('fusion', {}).get('out_dim', 1280)
        
        # ✅ 检测是否是 proprio 版 checkpoint
        policy_class = ckpt.get('policy_class', '')
        self.use_proprio = (policy_class == 'DirectFusionProprioPolicy')
        
        # 动作维度
        include_gripper = config.get('data', {}).get('include_gripper', True)
        if not include_gripper:
            raise ValueError("Checkpoint must have include_gripper=True for 14-dim actions")
        
        self.action_dim = 14  # 双臂+双夹爪
        # ✅ 训练时使用的动作范围（与DPRGBOnlineDataset一致）
        self.action_min = np.full(self.action_dim, -3.0, dtype=np.float32)
        self.action_max = np.full(self.action_dim, 3.0, dtype=np.float32)
        
        print(f"[DirectFusion] Config: horizon={self.horizon}, n_obs_steps={self.n_obs_steps}")
        print(f"[DirectFusion] Fusion: type={self.fusion_type}, dim={self.fuse_dim}")
        print(f"[DirectFusion] Policy class: {policy_class or 'DirectFusionDPPolicy'}")
        print(f"[DirectFusion] use_proprio: {self.use_proprio}")
        
        # 1. 加载4个RGB backbone
        print("[DirectFusion] Loading Vision Backbones...")
        # 支持通过 usr_args.gpu_ids 显式指定GPU (例如 --gpu_ids [0,1])
        requested_gpu_ids = self.usr_args.get("gpu_ids", None)
        if isinstance(requested_gpu_ids, (list, tuple)) and len(requested_gpu_ids) > 0:
            gpu_ids = list(requested_gpu_ids)
        else:
            if torch.cuda.device_count() > 1:
                other = 1 - self.gpu_id if self.gpu_id in (0, 1) else (self.gpu_id + 1) % torch.cuda.device_count()
                gpu_ids = [self.gpu_id, other]
            else:
                gpu_ids = [self.gpu_id]
        
        self.feature_extractors = MultiGPUFeatureExtractors(gpu_ids=gpu_ids)
        
        # 2. 创建融合编码器
        print("[DirectFusion] Creating Fusion Encoder...")
        fusion_encoder = SimpleFusionEncoder(
            in_dims=(1024, 2048, 768, 2048),
            fusion_type=self.fusion_type,
            out_dim=self.fuse_dim,
        )
        # 允许通过环境变量设置融合mask（默认全启用）
        mask_env = os.environ.get("DP2DP3_FUSION_MASK", "")
        if mask_env:
            try:
                mask_vals = [float(x.strip()) for x in mask_env.split(',')]
                if len(mask_vals) == 4:
                    fusion_encoder.fusion_mask = torch.tensor(mask_vals)
                    print(f"[DirectFusion] Using fusion mask: {mask_vals}")
                else:
                    print(f"[DirectFusion] Ignoring fusion mask (len!=4): {mask_env}")
            except Exception:
                print(f"[DirectFusion] Failed to parse fusion mask: {mask_env}")
        
        # 3. 创建Policy（根据 checkpoint 类型选择）
        print("[DirectFusion] Creating Policy...")
        if self.use_proprio:
            self.policy = DirectFusionProprioPolicy(
                fusion_encoder=fusion_encoder,
                proprio_dim=14,
                action_dim=self.action_dim,
                horizon=self.horizon,
                n_obs_steps=self.n_obs_steps,
                num_inference_steps=config.get('policy', {}).get('num_inference_steps', 100),
            )
        else:
            self.policy = DirectFusionDPPolicy(
                fusion_encoder=fusion_encoder,
                action_dim=self.action_dim,
                horizon=self.horizon,
                n_obs_steps=self.n_obs_steps,
                num_inference_steps=config.get('policy', {}).get('num_inference_steps', 100),
            )
        
        # 4. 加载权重
        print("[DirectFusion] Loading weights...")
        policy_state = ckpt.get('policy', {})
        
        # 处理可能的DDP包装
        if any(k.startswith('module.') for k in policy_state.keys()):
            policy_state = {k.replace('module.', ''): v for k, v in policy_state.items()}
        
        # 加载权重
        missing, unexpected = self.policy.load_state_dict(policy_state, strict=False)
        if missing:
            print(f"[WARNING] Missing keys: {missing}")
        if unexpected:
            print(f"[WARNING] Unexpected keys: {unexpected}")
        
        # 5. 加载normalizer
        if 'normalizer' in ckpt:
            print("[DirectFusion] Loading normalizer...")
            self.policy.normalizer.load_state_dict(ckpt['normalizer'])
            self.policy.use_normalizer = True
        else:
            print("[WARNING] Normalizer not found in checkpoint, outputs may be mis-scaled!")
            self.policy.use_normalizer = False

        try:
            self.policy.normalizer.to(self.device)
        except Exception:
            pass

        self.policy = self.policy.to(self.device).eval()
        
        print("[DirectFusion] All models loaded successfully!")

    def _manual_unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """当checkpoint没有normalizer时，使用固定范围将[-1,1]映射回真实动作"""
        action = np.clip(action, -1.0, 1.0)
        return (action + 1.0) * 0.5 * (self.action_max - self.action_min) + self.action_min
    
    def reset(self):
        """重置状态"""
        self.obs_buffer.clear()
        self.action_queue.clear()
    
    def get_action(self, obs):
        """获取动作"""
        # 添加观测到缓冲区
        self.obs_buffer.append(obs)
        
        # 填充缓冲区
        while len(self.obs_buffer) < self.n_obs_steps:
            self.obs_buffer.append(self.obs_buffer[0])
        
        # 如果动作队列为空，预测新的动作序列
        if self.replan_every_call or len(self.action_queue) == 0:
            # 1. 准备图像
            images = []
            for o in self.obs_buffer:
                img_np = (o['head_cam'].transpose(1, 2, 0) * 255.0).astype(np.uint8)
                img_np = np.clip(img_np, 0, 255)
                img_pil = Image.fromarray(img_np, mode='RGB')
                images.append(img_pil)
            
            # 1.5 ✅ 准备 agent_pos（如果是 proprio 模型）
            agent_pos_tensor = None
            if self.use_proprio:
                try:
                    ap_list = []
                    for o in self.obs_buffer:
                        ap = np.asarray(o.get('agent_pos', []), dtype=np.float32).reshape(-1)
                        ap_list.append(ap)
                    if len(ap_list) > 0 and all(a.shape == ap_list[0].shape for a in ap_list):
                        ap_seq = np.stack(ap_list, axis=0)  # [To, 14]
                        agent_pos_tensor = torch.from_numpy(ap_seq).float().to(self.device).unsqueeze(0)  # [1, To, 14]
                except Exception as e:
                    print(f"[WARNING] Failed to prepare agent_pos: {e}")
                    agent_pos_tensor = None
            
            # 2. 批量提取RGB特征
            # MultiGPUFeatureExtractors.extract_batch 返回 [B, 4, 2048]
            # 其中每个模型的特征都pad到2048维，实际维度是:
            # - CroCo: 1024维 (剩余pad 0)
            # - VGGT: 2048维 (完整)
            # - DINOv3: 768维 (剩余pad 0)  
            # - DA3: 2048维 (完整)
            features_np = self.feature_extractors.extract_batch(images)  # [To, 4, 2048]
            
            # 3. 转换为tensor并提取每个模型的实际维度
            # 我们需要 [1, To, M, C_i] 其中 C_i 是每个模型的真实维度
            # SimpleFusionEncoder的projections会处理不同的输入维度
            model_dims = [1024, 2048, 768, 2048]  # CroCo, VGGT, DINOv3, DA3
            To = features_np.shape[0]
            
            # 提取每个模型的真实维度特征（去掉padding）
            features_list = []
            for i, dim in enumerate(model_dims):
                feat = features_np[:, i, :dim]  # [To, dim_i]
                features_list.append(feat)
            
            # 为了适配SimpleFusionEncoder的forward，需要统一到 [B, To, M, C_max]
            # 但fusion encoder会对每个模型用不同的projection，所以需要保持原始维度
            # 我们需要修改SimpleFusionEncoder.forward的实现，或者在这里做特殊处理
            
            # 方案：直接传入 [B, To, M, C_max]，让fusion encoder处理
            # 注意：SimpleFusionEncoder的projections输入维度是固定的(1024, 2048, 768, 2048)
            # 所以我们需要确保传入的特征维度匹配
            
            # 重新构建：[To, M, C_i] -> [1, To, M, C_max] (padding到最大维度)
            max_dim = max(model_dims)
            features_padded = []
            for i, (feat, dim) in enumerate(zip(features_list, model_dims)):
                if dim < max_dim:
                    pad_width = max_dim - dim
                    feat_padded = np.pad(feat, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    feat_padded = feat
                features_padded.append(feat_padded)
            
            # Stack: [M, To, C_max] -> [To, M, C_max]
            features_np_unified = np.stack(features_padded, axis=1)  # [To, M, C_max]
            
            # 转tensor: [1, To, M, C_max]
            features = torch.from_numpy(features_np_unified).float().unsqueeze(0).to(self.device)
            
            # 4. 通过policy预测动作
            with torch.no_grad():
                if self.use_proprio:
                    action_pred = self.policy(features, agent_pos=agent_pos_tensor)
                else:
                    action_pred = self.policy(features)  # [1, Ta, A]
            
            action_pred = action_pred.squeeze(0).cpu().numpy()  # [Ta, A]

            # ✅ 如果没有normalizer，使用固定范围手动反归一化
            if not getattr(self.policy, 'use_normalizer', False):
                action_pred = self._manual_unnormalize_action(action_pred)
            
            # 🔍 调试输出: 模型输出范围（已注释）
            # print(f"[DEBUG] 模型输出:")
            # print(f"  Shape: {action_pred.shape}")
            # print(f"  Range: [{action_pred.min():.3f}, {action_pred.max():.3f}]")
            # print(f"  Mean: {action_pred.mean():.3f}, Std: {action_pred.std():.3f}")
            # print(f"  First action: {action_pred[0]}")

            # 🔧 安全限制：防止异常值
            action_pred_before_clip = action_pred.copy()
            action_pred = np.clip(action_pred, -3.0, 3.0)
            if not np.allclose(action_pred, action_pred_before_clip):
                print(f"[WARNING] 动作被Clip! 原始range: [{action_pred_before_clip.min():.3f}, {action_pred_before_clip.max():.3f}]")
            
            # 加入队列
            self.action_queue.clear()
            self.action_queue.extend(action_pred)
        
        # 🔧 修复: Receding Horizon - 执行更多步数以减少抖动
        # Horizon=8，执行前4步 (50%重叠)，保证轨迹连续性
        n_action_exec = min(self.n_action_exec, len(self.action_queue))
        return list(self.action_queue)[:n_action_exec]
    
    def update_obs(self, obs):
        """更新观测"""
        self.obs_buffer.append(obs)
    
    def pop_action(self):
        """弹出动作"""
        if len(self.action_queue) > 0:
            return self.action_queue.popleft()
        return np.zeros(self.action_dim, dtype=np.float32)


def get_model(usr_args):
    """RoBoTwin标准接口：获取模型"""
    model = DirectFusionModel(usr_args)
    return model


def eval(TASK_ENV, model, observation):
    """RoBoTwin标准接口：执行推理
    
    🔧 修复: obs/action同步问题
    - 问题: 每执行一个action就更新一次obs，导致缓冲区全是新观测，丢失历史
    - 修复: 只在执行完所有动作后更新一次观测，保持时序合理性
    """
    obs = encode_obs(observation)
    
    # 获取动作序列
    actions = model.get_action(obs)
    
    # 🔧 修复: 每步执行后更新观测，保持滑动窗口最新
    for i, action in enumerate(actions):
        TASK_ENV.take_action(action)
        model.pop_action()

        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    """RoBoTwin标准接口：重置模型"""
    model.reset()
