"""DP2DP3 Policy Deployment for RoBoTwin

这个文件是 DP2DP3 作为 RoBoTwin 标准 policy 的入口。
遵循 RoBoTwin 的 policy 接口规范（参考 DP/DP3）。

DP2DP3 架构：
1. RGB 图像 → 4 个视觉模型 (CroCo/VGGT/DINOv3/DA3) → RGB 特征
2. RGB 特征 → 对齐编码器 (RGB2PC) → 统一特征空间  
3. 统一特征 → Diffusion Policy Head → 动作输出

与 DP3 的区别：
- DP3: 直接使用点云 + ULIP
- DP2DP3: 使用 RGB 图像，通过蒸馏对齐到点云特征空间

接口要求（兼容 RoBoTwin 框架）：
- get_model(usr_args) -> Model 实例
- Model.encode_obs(observation) -> 编码观测
- eval(TASK_ENV, model, observation) -> 执行动作
- reset_model(model) -> 重置模型状态
"""

import sys
import os
import json
import time
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

from features_common.alignment.rgb2pc_aligned_encoder_4models import RGB2PCAlignedEncoder4Models
from features_common.multi_gpu_extractors import MultiGPUFeatureExtractors
from PIL import Image

# 导入正版DP
HAS_OFFICIAL_DP = False
try:
    # 添加正版DP外层路径（DP/diffusion_policy包含diffusion_policy子目录）
    DP_OUTER = Path(features_model_dir) / "DP" / "diffusion_policy"
    if DP_OUTER.exists() and (DP_OUTER / "diffusion_policy").exists():
        sys.path.insert(0, str(DP_OUTER))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        HAS_OFFICIAL_DP = True
        print("[INFO] 正版DP modules imported")
    else:
        print(f"[WARNING] 正版DP路径不存在: {DP_OUTER}")
except ImportError as e:
    print(f"[WARNING] 正版DP导入失败: {e}")


class DPRGBPolicy(nn.Module):
    """正版Diffusion Policy for RGB特征"""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int = 8,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        num_inference_steps: int = 100,
        use_proprio: bool = False,
        proprio_dim: int = 14,
        proprio_mode: str = "concat",
        proprio_hidden: int = 256,
    ):
        super().__init__()
        
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版DP未加载，无法使用DPRGBPolicy")
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        # ✅ 可选normalizer（若checkpoint包含则启用）
        self.normalizer = LinearNormalizer()
        self.use_normalizer = False
        
        # 观测编码器（纯视觉）
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        # 可选：本体感知编码器（不改变视觉对齐模块）
        self.use_proprio = bool(use_proprio)
        self.proprio_mode = str(proprio_mode)
        if self.use_proprio:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(int(proprio_dim) * int(n_obs_steps), proprio_hidden),
                nn.ReLU(),
                nn.Linear(proprio_hidden, 256),
                nn.ReLU(),
            )
            if self.proprio_mode == "concat":
                self.proprio_fuse = nn.Sequential(
                    nn.Linear(256 + 256, 256),
                    nn.ReLU(),
                )
            elif self.proprio_mode != "add":
                print(f"[Warning] unknown proprio_mode={self.proprio_mode}, fallback to 'concat'")
                self.proprio_mode = "concat"
                self.proprio_fuse = nn.Sequential(
                    nn.Linear(256 + 256, 256),
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

    def forward(self, obs, agent_pos=None):
        """推理模式"""
        B = obs.shape[0]
        device = obs.device
        
        # 编码观测
        obs_flat = obs.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        if self.use_proprio and agent_pos is not None:
            if self.use_normalizer and hasattr(self, "normalizer") and "agent_pos" in self.normalizer.params_dict:
                agent_pos = self.normalizer["agent_pos"].normalize(agent_pos)
            proprio_flat = agent_pos.reshape(B, -1)
            proprio_cond = self.proprio_encoder(proprio_flat)
            if self.proprio_mode == "add":
                obs_cond = obs_cond + proprio_cond
            else:
                obs_cond = self.proprio_fuse(torch.cat([obs_cond, proprio_cond], dim=-1))
        
        # 初始化随机噪声
        action = torch.randn((B, self.n_action_steps, self.action_dim), device=device)
        
        # 设置推理步数
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        # 逐步去噪
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(B).to(device),
                global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample
        
        if self.use_normalizer:
            action = self.normalizer.unnormalize({'action': action})['action']
        return action


class DPRGBDualStreamPolicy(nn.Module):
    """Dual-stream DP policy with token cross-attention conditioning."""

    def __init__(
        self,
        obs_dim: int,
        token_dim: int,
        action_dim: int,
        horizon: int = 8,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        num_inference_steps: int = 100,
        token_dropout: float = 0.0,
        ctx_dropout: float = 0.0,
        token_gate_init: float = -4.0,
        use_proprio: bool = False,
        proprio_dim: int = 14,
        proprio_mode: str = "concat",
        proprio_hidden: int = 256,
    ):
        super().__init__()

        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版DP未加载，无法使用DPRGBDualStreamPolicy")

        self.obs_dim = obs_dim
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        self.normalizer = LinearNormalizer()
        self.use_normalizer = False

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.use_proprio = bool(use_proprio)
        self.proprio_mode = str(proprio_mode)
        if self.use_proprio:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(int(proprio_dim) * int(n_obs_steps), proprio_hidden),
                nn.ReLU(),
                nn.Linear(proprio_hidden, 256),
                nn.ReLU(),
            )
            if self.proprio_mode == "concat":
                self.proprio_fuse = nn.Sequential(
                    nn.Linear(256 + 256, 256),
                    nn.ReLU(),
                )
            elif self.proprio_mode != "add":
                print(f"[Warning] unknown proprio_mode={self.proprio_mode}, fallback to 'concat'")
                self.proprio_mode = "concat"
                self.proprio_fuse = nn.Sequential(
                    nn.Linear(256 + 256, 256),
                    nn.ReLU(),
                )
        self.token_proj = nn.Linear(token_dim, 256)
        self.query_proj = nn.Linear(256, 256)
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        self.token_dropout = nn.Dropout(float(token_dropout)) if float(token_dropout) > 0 else nn.Identity()
        self.ctx_dropout = nn.Dropout(float(ctx_dropout)) if float(ctx_dropout) > 0 else nn.Identity()
        self.token_gate = nn.Parameter(torch.tensor(float(token_gate_init)))
        # 可选：推理期强制gate值（用于与训练warmup阶段对齐）
        self.force_gate = None

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

    def _build_cond(self, obs_global: torch.Tensor, obs_tokens: torch.Tensor, agent_pos: torch.Tensor = None) -> torch.Tensor:
        b = obs_global.shape[0]
        obs_flat = obs_global.reshape(b, -1)
        global_cond = self.obs_encoder(obs_flat)
        if self.use_proprio and agent_pos is not None:
            if self.use_normalizer and hasattr(self, "normalizer") and "agent_pos" in self.normalizer.params_dict:
                agent_pos = self.normalizer["agent_pos"].normalize(agent_pos)
            proprio_flat = agent_pos.reshape(b, -1)
            proprio_cond = self.proprio_encoder(proprio_flat)
            if self.proprio_mode == "add":
                global_cond = global_cond + proprio_cond
            else:
                global_cond = self.proprio_fuse(torch.cat([global_cond, proprio_cond], dim=-1))
        tokens = obs_tokens.reshape(b, -1, obs_tokens.shape[-1])
        tokens = self.token_proj(tokens)
        tokens = self.token_dropout(tokens)
        query = self.query_proj(global_cond).unsqueeze(1)
        ctx, _ = self.cross_attn(query, tokens, tokens)
        ctx = self.ctx_dropout(ctx.squeeze(1))
        if self.force_gate is not None:
            gate = torch.tensor(float(self.force_gate), device=ctx.device, dtype=ctx.dtype)
        else:
            gate = torch.sigmoid(self.token_gate)
        return global_cond + gate * ctx

    def forward(self, obs_global: torch.Tensor, obs_tokens: torch.Tensor, agent_pos: torch.Tensor = None):
        b = obs_global.shape[0]
        device = obs_global.device
        obs_cond = self._build_cond(obs_global, obs_tokens, agent_pos)

        action = torch.randn((b, self.n_action_steps, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action,
                t.unsqueeze(0).expand(b).to(device),
                global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        if self.use_normalizer:
            action = self.normalizer.unnormalize({'action': action})['action']
        return action


class DPAlignedPolicy(nn.Module):
    """
    DP-Aligned Policy: agent_pos 直接 concat 到视觉特征（与原版 DP 一致）

    与 DPRGBPolicy 的区别：
    - agent_pos 不通过独立 MLP，而是归一化后直接拼接到 obs 特征
    - obs_encoder 输入维度 = n_obs_steps * (effective_vis_dim + proprio_dim)
    - 支持可选的 vis_projector（方案A：对齐特征后加可学习投影层）
    """

    def __init__(
        self,
        vis_dim: int,
        action_dim: int,
        horizon: int = 8,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        num_inference_steps: int = 100,
        use_proprio: bool = False,
        proprio_dim: int = 14,
        vis_projector_type: str = "none",
        vis_projector_dim: int = 1280,
    ):
        super().__init__()
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版DP未加载")

        self.vis_dim = vis_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        self.use_proprio = bool(use_proprio)
        self.proprio_dim = int(proprio_dim) if use_proprio else 0
        self.normalizer = LinearNormalizer()
        self.use_normalizer = False

        # 可选的可学习 vis_projector
        self.vis_projector_type = vis_projector_type
        if vis_projector_type == "linear":
            self.vis_projector = nn.Sequential(
                nn.Linear(vis_dim, vis_projector_dim),
                nn.LayerNorm(vis_projector_dim),
            )
            effective_vis_dim = vis_projector_dim
        elif vis_projector_type == "mlp":
            self.vis_projector = nn.Sequential(
                nn.Linear(vis_dim, vis_dim * 2),
                nn.GELU(),
                nn.Linear(vis_dim * 2, vis_projector_dim),
                nn.LayerNorm(vis_projector_dim),
            )
            effective_vis_dim = vis_projector_dim
        else:
            self.vis_projector = None
            effective_vis_dim = vis_dim

        per_step_dim = effective_vis_dim + (self.proprio_dim if use_proprio else 0)
        obs_input_dim = n_obs_steps * per_step_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim, 512),
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

    def forward(self, obs, agent_pos=None):
        B = obs.shape[0]
        device = obs.device

        # 如果有可学习的 vis_projector，先对视觉特征做投影
        if self.vis_projector is not None:
            B_orig, To, D = obs.shape
            obs = self.vis_projector(obs.reshape(B_orig * To, D)).reshape(B_orig, To, -1)

        if self.use_proprio and agent_pos is not None:
            if self.use_normalizer and "agent_pos" in self.normalizer.params_dict:
                agent_pos = self.normalizer["agent_pos"].normalize(agent_pos)
            obs_combined = torch.cat([obs, agent_pos], dim=-1)
        else:
            obs_combined = obs
        obs_flat = obs_combined.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)

        action = torch.randn((B, self.n_action_steps, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action, t.unsqueeze(0).expand(B).to(device), global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        if self.use_normalizer:
            action = self.normalizer.unnormalize({'action': action})['action']
        return action


class DPAlignedDualStreamPolicy(nn.Module):
    """
    DP-Aligned Dual-Stream Policy: agent_pos 直接 concat + token cross-attention
    """

    def __init__(
        self,
        vis_dim: int,
        token_dim: int,
        action_dim: int,
        horizon: int = 8,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        num_inference_steps: int = 100,
        token_dropout: float = 0.0,
        ctx_dropout: float = 0.0,
        token_gate_init: float = -4.0,
        use_proprio: bool = False,
        proprio_dim: int = 14,
    ):
        super().__init__()
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版DP未加载")

        self.vis_dim = vis_dim
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        self.use_proprio = bool(use_proprio)
        self.proprio_dim = int(proprio_dim) if use_proprio else 0
        self.normalizer = LinearNormalizer()
        self.use_normalizer = False

        per_step_dim = vis_dim + (self.proprio_dim if use_proprio else 0)
        obs_input_dim = n_obs_steps * per_step_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.token_proj = nn.Linear(token_dim, 256)
        self.query_proj = nn.Linear(256, 256)
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        self.token_dropout = nn.Dropout(float(token_dropout)) if float(token_dropout) > 0 else nn.Identity()
        self.ctx_dropout = nn.Dropout(float(ctx_dropout)) if float(ctx_dropout) > 0 else nn.Identity()
        self.token_gate = nn.Parameter(torch.tensor(float(token_gate_init)))
        self.force_gate = None

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

    def _build_cond(self, obs_global, obs_tokens, agent_pos=None):
        B = obs_global.shape[0]
        if self.use_proprio and agent_pos is not None:
            if self.use_normalizer and "agent_pos" in self.normalizer.params_dict:
                agent_pos = self.normalizer["agent_pos"].normalize(agent_pos)
            obs_combined = torch.cat([obs_global, agent_pos], dim=-1)
        else:
            obs_combined = obs_global
        obs_flat = obs_combined.reshape(B, -1)
        global_cond = self.obs_encoder(obs_flat)

        tokens = obs_tokens.reshape(B, -1, obs_tokens.shape[-1])
        tokens = self.token_proj(tokens)
        tokens = self.token_dropout(tokens)
        query = self.query_proj(global_cond).unsqueeze(1)
        ctx, _ = self.cross_attn(query, tokens, tokens)
        ctx = self.ctx_dropout(ctx.squeeze(1))
        if self.force_gate is not None:
            gate = torch.tensor(float(self.force_gate), device=ctx.device, dtype=ctx.dtype)
        else:
            gate = torch.sigmoid(self.token_gate)
        return global_cond + gate * ctx

    def forward(self, obs_global, obs_tokens, agent_pos=None):
        B = obs_global.shape[0]
        device = obs_global.device
        obs_cond = self._build_cond(obs_global, obs_tokens, agent_pos)

        action = torch.randn((B, self.n_action_steps, self.action_dim), device=device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                action, t.unsqueeze(0).expand(B).to(device), global_cond=obs_cond
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        if self.use_normalizer:
            action = self.normalizer.unnormalize({'action': action})['action']
        return action


def encode_obs(observation):
    """
    将 RoBoTwin 环境的观测转换为 DP2DP3 所需的格式
    
    Args:
        observation: RoBoTwin 环境返回的观测字典
            - 'joint_action': {'vector': ...} - 机器人关节状态
            - 'observation': {'head_camera': {'rgb': ...}} - RGB 图像
            
    Returns:
        obs: 处理后的观测字典
            - 'agent_pos': 关节状态
            - 'head_cam': RGB 图像（归一化到 [0,1]）
    """
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    
    # DP2DP3 使用 RGB 图像
    head_cam = observation["observation"]["head_camera"]["rgb"]
    # 转换为 [C, H, W] 格式并归一化
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0
    obs['head_cam'] = head_cam
    
    return obs


class DP2DP3Model:
    """
    DP2DP3 Policy Model Wrapper for RoBoTwin
    
    这个类封装了完整的 DP2DP3 推理流程：
    1. 从 checkpoint 加载对齐编码器和动作头
    2. 实时提取 RGB 特征（通过 4 个视觉模型）
    3. 通过对齐编码器和 Diffusion Head 预测动作
    """
    
    def __init__(self, usr_args):
        """
        初始化 DP2DP3 模型
        
        Args:
            usr_args: 用户参数字典，包含：
                - task_name: 任务名称
                - ckpt_setting: checkpoint 设置
                - expert_data_num: 专家数据数量
                - checkpoint_num: checkpoint 编号
                - seed: 随机种子
        """
        self.usr_args = usr_args
        # GPU ID: 在 CUDA_VISIBLE_DEVICES 环境下始终使用 0（已经被映射）
        self.gpu_id = 0
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")

        # 轻量动作后处理：降低抖动
        self.action_smoothing_alpha = float(usr_args.get('action_smoothing_alpha', 0.20))
        # NOTE: 用 Optional 语义即可，避免 py<3.10 对 `|` 的解析问题
        self._prev_action = None  # type: np.ndarray | None
        # 是否每次 get_action 都重规划（默认 True，保持 baseline 行为）
        replan_every_call = usr_args.get("replan_every_call", os.environ.get("DP2DP3_REPLAN_EVERY_CALL", "1"))
        try:
            self.replan_every_call = bool(int(replan_every_call)) if isinstance(replan_every_call, (str, int)) else bool(replan_every_call)
        except Exception:
            self.replan_every_call = True
        self.n_action_exec = int(usr_args.get('n_action_exec', 4))

        # 调试开关：默认关闭，避免评估时大量 stdout 影响速度/可读性
        self.debug_actions = bool(int(os.environ.get("DP2DP3_DEBUG_ACTIONS", "0")))
        
        # 模型路径配置
        self.features_model_dir = Path(features_model_dir)
        self.policy_dir = Path(policy_dir)
        self.task_name = usr_args['task_name']
        self.ckpt_setting = usr_args.get('ckpt_setting', 'demo_randomized')
        self.expert_data_num = usr_args.get('expert_data_num', 20)
        self.seed = usr_args.get('seed', 0)
        self.checkpoint_num = usr_args.get('checkpoint_num', 50)
        
        # 构建标准化 checkpoint 路径
        # 格式: /home/gl/RoboTwin/policy/DP2DP3/checkpoints/{task}-{setting}-{num}-{seed}/{epoch}.ckpt
        ckpt_dir_name = f"{self.task_name}-{self.ckpt_setting}-{self.expert_data_num}-{self.seed}"
        standard_ckpt_dir = self.policy_dir / "checkpoints" / ckpt_dir_name
        
        if str(self.checkpoint_num) == 'best' or str(self.checkpoint_num).lower() == 'best':
            # 查找最新的 checkpoint
            ckpt_files = list(standard_ckpt_dir.glob("*.ckpt"))
            if ckpt_files:
                self.head_ckpt_path = max(ckpt_files, key=lambda p: int(p.stem))
                print(f"[DP2DP3] Using 'best' checkpoint (latest)")
            else:
                raise FileNotFoundError(f"No checkpoint found in {standard_ckpt_dir}")
        else:
            standard_ckpt_path = standard_ckpt_dir / f"{int(self.checkpoint_num)}.ckpt"
            if standard_ckpt_path.exists():
                self.head_ckpt_path = standard_ckpt_path
            else:
                raise FileNotFoundError(f"Checkpoint not found: {standard_ckpt_path}")
        
        print(f"[DP2DP3] Loading checkpoint: {self.head_ckpt_path}")
        
        # 加载模型
        self._load_models()
        
        # 观测缓冲区（用于滑动窗口）
        self.obs_buffer = deque(maxlen=self.n_obs_steps)
        
        # 🔧 优化：特征缓存（避免重复提取）
        self.feature_cache = deque(maxlen=self.n_obs_steps)
        
        # 动作队列（用于 receding horizon）
        self.action_queue = deque()

        # 可选：动作日志输出
        self.action_log_path = usr_args.get('action_log_path', None)
        self._action_log_initialized = False
        self._action_log_plan_id = 0
        self._action_log_step_id = 0
        self._action_log_current_plan_id = None

        # 可选：固定token采样随机种子（减少随机token导致的动作抖动）
        self.token_sample_seed = usr_args.get('token_sample_seed', None)
        # token采样模式: random | deterministic
        self.token_sample_mode = str(usr_args.get('token_sample_mode', 'random')).lower()

        # 可选：初始化 prev_action（默认关闭，保证不影响 baseline）
        # 目标：解决“第一秒就瞬移”的一类问题（episode 开始时 prev_action=None 导致 step_delta_clamp 不生效）。
        # 方法：把当前观测的关节状态（agent_pos）作为 prev_action，从而把第一个执行动作也纳入 rate limiter。
        init_prev_action_from_obs = usr_args.get(
            "init_prev_action_from_obs", os.environ.get("DP2DP3_INIT_PREV_ACTION_FROM_OBS", "0")
        )
        try:
            self.init_prev_action_from_obs = bool(int(init_prev_action_from_obs)) if isinstance(init_prev_action_from_obs, (str, int)) else bool(init_prev_action_from_obs)
        except Exception:
            self.init_prev_action_from_obs = False

        # 可选：Chunk 边界动作一致性约束（默认关闭，保证不影响 baseline）
        # 目标：减少重规划时由于扩散采样随机性导致的“闪现/横跳”大跳变
        boundary_clamp = usr_args.get("boundary_clamp", os.environ.get("DP2DP3_BOUNDARY_CLAMP", "0"))
        try:
            self.boundary_clamp = bool(int(boundary_clamp)) if isinstance(boundary_clamp, (str, int)) else bool(boundary_clamp)
        except Exception:
            self.boundary_clamp = False
        boundary_max_delta = usr_args.get("boundary_max_delta", os.environ.get("DP2DP3_BOUNDARY_MAX_DELTA", "1.0"))
        try:
            self.boundary_max_delta = float(boundary_max_delta)
        except Exception:
            self.boundary_max_delta = 1.0

        boundary_clamp_steps = usr_args.get(
            "boundary_clamp_steps", os.environ.get("DP2DP3_BOUNDARY_CLAMP_STEPS", "1")
        )
        try:
            self.boundary_clamp_steps = int(boundary_clamp_steps)
        except Exception:
            self.boundary_clamp_steps = 1

        # 可选：逐步动作变化率限制（默认关闭）
        # 目标：把“瞬移/闪现”变成可执行的平滑轨迹，类似关节速度上限（rate limiter）。
        step_delta_clamp = usr_args.get("step_delta_clamp", os.environ.get("DP2DP3_STEP_DELTA_CLAMP", "0"))
        try:
            self.step_delta_clamp = bool(int(step_delta_clamp)) if isinstance(step_delta_clamp, (str, int)) else bool(step_delta_clamp)
        except Exception:
            self.step_delta_clamp = False
        step_max_delta = usr_args.get("step_max_delta", os.environ.get("DP2DP3_STEP_MAX_DELTA", "1.0"))
        try:
            self.step_max_delta = float(step_max_delta)
        except Exception:
            self.step_max_delta = 1.0
        step_delta_clamp_steps = usr_args.get(
            "step_delta_clamp_steps", os.environ.get("DP2DP3_STEP_DELTA_CLAMP_STEPS", "1")
        )
        try:
            self.step_delta_clamp_steps = int(step_delta_clamp_steps)
        except Exception:
            self.step_delta_clamp_steps = 1

        # 可选：重规划时的“计划重叠融合”（默认关闭）
        # 目标：抑制“反复来回/回跳”——当每次重规划都从随机噪声采样时，新 plan 往往与上一段剩余 plan 不一致，
        # 会导致 robot 一会儿靠近、一会儿撤退。融合用上一段剩余动作作为锚点，要求新 plan 的前缀更连续。
        overlap_blend = usr_args.get("overlap_blend", os.environ.get("DP2DP3_OVERLAP_BLEND", "0"))
        try:
            self.overlap_blend = bool(int(overlap_blend)) if isinstance(overlap_blend, (str, int)) else bool(overlap_blend)
        except Exception:
            self.overlap_blend = False
        overlap_blend_alpha = usr_args.get("overlap_blend_alpha", os.environ.get("DP2DP3_OVERLAP_BLEND_ALPHA", "0.2"))
        try:
            self.overlap_blend_alpha = float(overlap_blend_alpha)
        except Exception:
            self.overlap_blend_alpha = 0.2
        overlap_blend_steps = usr_args.get("overlap_blend_steps", os.environ.get("DP2DP3_OVERLAP_BLEND_STEPS", "0"))
        try:
            self.overlap_blend_steps = int(overlap_blend_steps)
        except Exception:
            self.overlap_blend_steps = 0

        # ✅ 关键修复：Best-of-N 计划采样 + 连续性选择（默认关闭，保证不影响81%基线）
        # 背景：token_full 的 diffusion head 在每次重规划都会从随机噪声重新采样一条 horizon 轨迹，
        # 新 plan 可能与上一段剩余 plan 完全不一致，出现“回跳/反复来回”。
        # 方案：同一观测下采样 N 条候选 plan，用“与上一段剩余动作的重叠一致性”作为主要评分，
        # 选出最连续的一条，而不是直接用单次采样结果或做线性融合（融合容易产生奇怪中间态）。
        plan_select_n = usr_args.get("plan_select_n", os.environ.get("DP2DP3_PLAN_SELECT_N", "1"))
        try:
            self.plan_select_n = max(1, int(plan_select_n))
        except Exception:
            self.plan_select_n = 1
        plan_select_overlap_steps = usr_args.get(
            "plan_select_overlap_steps", os.environ.get("DP2DP3_PLAN_SELECT_OVERLAP_STEPS", "0")
        )
        try:
            self.plan_select_overlap_steps = max(0, int(plan_select_overlap_steps))
        except Exception:
            self.plan_select_overlap_steps = 0
        plan_select_w_overlap = usr_args.get("plan_select_w_overlap", os.environ.get("DP2DP3_PLAN_SELECT_W_OVERLAP", "1.0"))
        plan_select_w_first = usr_args.get("plan_select_w_first", os.environ.get("DP2DP3_PLAN_SELECT_W_FIRST", "0.05"))
        plan_select_w_smooth = usr_args.get("plan_select_w_smooth", os.environ.get("DP2DP3_PLAN_SELECT_W_SMOOTH", "0.02"))
        plan_select_w_cut = usr_args.get("plan_select_w_cut", os.environ.get("DP2DP3_PLAN_SELECT_W_CUT", "0.2"))
        try:
            self.plan_select_w_overlap = float(plan_select_w_overlap)
        except Exception:
            self.plan_select_w_overlap = 1.0
        try:
            self.plan_select_w_first = float(plan_select_w_first)
        except Exception:
            self.plan_select_w_first = 0.05
        try:
            self.plan_select_w_smooth = float(plan_select_w_smooth)
        except Exception:
            self.plan_select_w_smooth = 0.02
        try:
            self.plan_select_w_cut = float(plan_select_w_cut)
        except Exception:
            self.plan_select_w_cut = 0.2
        plan_select_seed_base = usr_args.get("plan_select_seed_base", os.environ.get("DP2DP3_PLAN_SELECT_SEED_BASE", "0"))
        try:
            self.plan_select_seed_base = int(plan_select_seed_base)
        except Exception:
            self.plan_select_seed_base = 0

        # plan_select 模式：
        # - "guard" (默认)：以候选0作为“baseline plan”，仅在与上一段剩余动作差异过大时才启用 Best-of-N 选择，
        #   目标：显著降低回跳，同时尽量不牺牲成功率/速度。
        # - "min_cost"：对 N 个候选全量打分，选总 cost 最小者（更激进，可能更慢/更保守）。
        self.plan_select_mode = str(
            usr_args.get("plan_select_mode", os.environ.get("DP2DP3_PLAN_SELECT_MODE", "guard"))
        ).strip().lower()

        # guard 阈值：用于判断“候选0 与上一段剩余动作”是否差异过大（默认较宽松）
        plan_select_guard_overlap_mse = usr_args.get(
            "plan_select_guard_overlap_mse", os.environ.get("DP2DP3_PLAN_SELECT_GUARD_OVERLAP_MSE", "0.05")
        )
        plan_select_guard_overlap_maxabs = usr_args.get(
            "plan_select_guard_overlap_maxabs", os.environ.get("DP2DP3_PLAN_SELECT_GUARD_OVERLAP_MAXABS", "0.5")
        )
        try:
            self.plan_select_guard_overlap_mse = float(plan_select_guard_overlap_mse)
        except Exception:
            self.plan_select_guard_overlap_mse = 0.05
        try:
            self.plan_select_guard_overlap_maxabs = float(plan_select_guard_overlap_maxabs)
        except Exception:
            self.plan_select_guard_overlap_maxabs = 0.5

        # 额外 guard：关注 “cut 点”(K-1 -> K) 的大跳变。K=n_action_exec。
        # 该跳变会在下一次重规划边界显性表现为 TCP 大跳/回跳。
        plan_select_guard_cut_maxabs = usr_args.get(
            "plan_select_guard_cut_maxabs", os.environ.get("DP2DP3_PLAN_SELECT_GUARD_CUT_MAXABS", "0.8")
        )
        try:
            self.plan_select_guard_cut_maxabs = float(plan_select_guard_cut_maxabs)
        except Exception:
            self.plan_select_guard_cut_maxabs = 0.8

        # 可选：对“强不一致”的边界做 ramp-blend（默认关闭；开启可能影响成功率）
        plan_select_ramp_blend = usr_args.get(
            "plan_select_ramp_blend", os.environ.get("DP2DP3_PLAN_SELECT_RAMP_BLEND", "0")
        )
        try:
            self.plan_select_ramp_blend = bool(int(plan_select_ramp_blend)) if isinstance(plan_select_ramp_blend, (str, int)) else bool(plan_select_ramp_blend)
        except Exception:
            self.plan_select_ramp_blend = False

        # “回跳”防护：如果新 plan 的第1步动作非常接近过去(10~80步前)某个动作，则极可能出现“回到几秒前姿态重来”
        # 该防护只用于 plan_select_n>1 时的候选选择，不改变 baseline。
        rewind_guard = usr_args.get("rewind_guard", os.environ.get("DP2DP3_REWIND_GUARD", "0"))
        try:
            self.rewind_guard = bool(int(rewind_guard)) if isinstance(rewind_guard, (str, int)) else bool(rewind_guard)
        except Exception:
            self.rewind_guard = True
        rewind_guard_window_min = usr_args.get(
            "rewind_guard_window_min", os.environ.get("DP2DP3_REWIND_GUARD_WINDOW_MIN", "10")
        )
        rewind_guard_window_max = usr_args.get(
            "rewind_guard_window_max", os.environ.get("DP2DP3_REWIND_GUARD_WINDOW_MAX", "80")
        )
        rewind_guard_thresh = usr_args.get(
            "rewind_guard_thresh", os.environ.get("DP2DP3_REWIND_GUARD_THRESH", "0.05")
        )
        try:
            self.rewind_guard_window_min = int(rewind_guard_window_min)
        except Exception:
            self.rewind_guard_window_min = 10
        try:
            self.rewind_guard_window_max = int(rewind_guard_window_max)
        except Exception:
            self.rewind_guard_window_max = 80
        try:
            self.rewind_guard_thresh = float(rewind_guard_thresh)
        except Exception:
            self.rewind_guard_thresh = 0.05

        # 记录已执行动作历史（每 episode 重置），用于回跳检测
        self._exec_action_hist: deque = deque(maxlen=512)

        # 用于按 episode 切分日志（便于量化“闪现/回跳”是否发生在重规划边界/episode 起始）
        # 注意：reset() 会自增 episode_id；第一次 reset 后 episode_id=0。
        self._episode_id = -1

        print(f"[DP2DP3] Model initialization complete")
        print(f"  Task: {self.task_name}")
        print(f"  Horizon: {self.horizon}, N_obs_steps: {self.n_obs_steps}")
        print(f"  Action dim: {self.action_dim}")

    def _predict_action_sequence(
        self,
        obs_emb: torch.Tensor,
        obs_tokens: torch.Tensor | None,
        agent_pos: torch.Tensor | None = None,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        采样一条动作序列（horizon长度），并做必要的反归一化与clip。
        注意：这里只做“不会改变策略语义”的最小后处理，用于 Best-of-N 选择。
        """
        use_cuda = torch.cuda.is_available()
        fork_devices = [int(self.gpu_id)] if use_cuda else []

        def _forward() -> torch.Tensor:
            if isinstance(self.policy, (DPRGBDualStreamPolicy, DPAlignedDualStreamPolicy)):
                if obs_tokens is None:
                    raise RuntimeError("DualStream policy requires token features, but obs_tokens is None")
                return self.policy(obs_emb, obs_tokens, agent_pos=agent_pos)
            return self.policy(obs_emb, agent_pos=agent_pos)

        with torch.no_grad():
            if seed is None:
                action_t = _forward()
            else:
                with torch.random.fork_rng(devices=fork_devices, enabled=True):
                    s = int(seed)
                    torch.manual_seed(s)
                    if use_cuda:
                        try:
                            torch.cuda.manual_seed_all(s)
                        except Exception:
                            pass
                    action_t = _forward()

        action_pred = action_t.squeeze(0).detach().cpu().numpy()  # [Ta, A]

        # 归一化反变换（若policy没有内置normalizer，才使用action_stats）
        if not getattr(self.policy, "use_normalizer", False) and self.action_stats is not None:
            action_min = np.array(self.action_stats.get("min", -3.0), dtype=np.float32)
            action_max = np.array(self.action_stats.get("max", 3.0), dtype=np.float32)
            action_pred = (action_pred + 1.0) * 0.5 * (action_max - action_min) + action_min

        # 安全限制：防止异常值
        action_pred = np.clip(action_pred, -3.0, 3.0)
        return action_pred.astype(np.float32, copy=False)

    def _append_action_log(self, payload: dict) -> None:
        if not self.action_log_path:
            return
        try:
            log_path = Path(self.action_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._action_log_initialized:
                meta = {
                    "stage": "meta",
                    "time": time.time(),
                    "task": self.task_name,
                    "ckpt": str(self.head_ckpt_path),
                    "horizon": int(self.horizon),
                    "n_obs_steps": int(self.n_obs_steps),
                    "n_action_exec": int(self.n_action_exec),
                    "use_token_infer": bool(self.use_token_infer),
                    "plan_select_n": int(getattr(self, "plan_select_n", 1)),
                    "plan_select_mode": str(getattr(self, "plan_select_mode", "guard")),
                    "plan_select_overlap_steps": int(getattr(self, "plan_select_overlap_steps", 0)),
                    "plan_select_w_overlap": float(getattr(self, "plan_select_w_overlap", 1.0)),
                    "plan_select_w_first": float(getattr(self, "plan_select_w_first", 0.05)),
                    "plan_select_w_smooth": float(getattr(self, "plan_select_w_smooth", 0.02)),
                    "plan_select_w_cut": float(getattr(self, "plan_select_w_cut", 0.2)),
                    "plan_select_seed_base": int(getattr(self, "plan_select_seed_base", 0)),
                    "plan_select_guard_overlap_mse": float(getattr(self, "plan_select_guard_overlap_mse", 0.05)),
                    "plan_select_guard_overlap_maxabs": float(getattr(self, "plan_select_guard_overlap_maxabs", 0.5)),
                    "plan_select_guard_cut_maxabs": float(getattr(self, "plan_select_guard_cut_maxabs", 0.8)),
                    "plan_select_ramp_blend": bool(getattr(self, "plan_select_ramp_blend", False)),
                    "rewind_guard": bool(getattr(self, "rewind_guard", True)),
                    "rewind_guard_window_min": int(getattr(self, "rewind_guard_window_min", 10)),
                    "rewind_guard_window_max": int(getattr(self, "rewind_guard_window_max", 80)),
                    "rewind_guard_thresh": float(getattr(self, "rewind_guard_thresh", 0.05)),
                    "boundary_clamp": bool(self.boundary_clamp),
                    "boundary_max_delta": float(self.boundary_max_delta),
                    "boundary_clamp_steps": int(self.boundary_clamp_steps),
                    "step_delta_clamp": bool(self.step_delta_clamp),
                    "step_max_delta": float(self.step_max_delta),
                    "step_delta_clamp_steps": int(self.step_delta_clamp_steps),
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                self._action_log_initialized = True
            payload = dict(payload)
            payload.setdefault("time", time.time())
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[DP2DP3] Action log write failed: {e}")
        
    def _load_models(self):
        """加载对齐编码器和动作头"""
        ckpt = torch.load(self.head_ckpt_path, map_location=self.device)
        self.action_stats = ckpt.get('action_stats') or ckpt.get('action_norm_stats')
        
        # 从 checkpoint 恢复配置
        config = ckpt.get('config', {})
        self.horizon = config.get('data', {}).get('horizon', 8)
        self.n_obs_steps = config.get('data', {}).get('n_obs_steps', 2)
        # 默认使用双臂，计算 action_dim
        self.action_dim = 0
        use_left = config.get('data', {}).get('use_left_arm', True)
        use_right = config.get('data', {}).get('use_right_arm', True)
        # include_gripper 默认为 False? 需检查 checkpoint config
        include_gripper = bool(config.get('data', {}).get('include_gripper', False))
        
        # RoboTwin take_action 期望动作格式：
        # [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] => 总 14 维
        self.env_left_arm_dim = int(self.usr_args.get('left_arm_dim', 6))
        self.env_right_arm_dim = int(self.usr_args.get('right_arm_dim', 6))
        self.env_action_dim = self.env_left_arm_dim + 1 + self.env_right_arm_dim + 1
        self.action_dim = self.env_action_dim

        # 强约束：部署期要求模型本身就输出包含 gripper 的 14 维动作。
        # 之前对 12 维动作做补零会导致夹爪完全不动，且行为很差。
        if not include_gripper:
            raise ValueError(
                "Checkpoint config data.include_gripper=False：该 head 训练时未监督夹爪维度。"
                "请用 include_gripper=true 重新训练 head，使其直接输出 14 维。"
            )

        # 进一步：如果 checkpoint 的 head 明确给出了输出维度，用它来推断 action_dim，避免维度不匹配
        policy_state = ckpt.get('policy', {})
        if isinstance(policy_state, dict):
            out_w = policy_state.get('net.4.weight')
            out_b = policy_state.get('net.4.bias')
            if out_w is not None and hasattr(out_w, 'shape'):
                out_dim = int(out_w.shape[0])
                if self.horizon > 0 and out_dim % int(self.horizon) == 0:
                    inferred_action_dim = int(out_dim // int(self.horizon))
                    self.model_action_dim = inferred_action_dim
            elif out_b is not None and hasattr(out_b, 'shape'):
                out_dim = int(out_b.shape[0])
                if self.horizon > 0 and out_dim % int(self.horizon) == 0:
                    inferred_action_dim = int(out_dim // int(self.horizon))
                    self.model_action_dim = inferred_action_dim

        # 如果 checkpoint 没有提供输出维度，就假设 head 直接输出 env_action_dim
        if not hasattr(self, 'model_action_dim'):
            self.model_action_dim = self.env_action_dim

        # 再做一次硬校验：必须等于 env_action_dim
        if int(self.model_action_dim) != int(self.env_action_dim):
            raise ValueError(
                f"Model action dim ({self.model_action_dim}) != env expected ({self.env_action_dim}). "
                "请确认 head 训练输出维度为 14（双臂+双夹爪）。"
            )
        
        print(f"[DP2DP3] Checkpoint keys: {list(ckpt.keys())}")
        print(f"[DP2DP3] Config: horizon={self.horizon}, n_obs_steps={self.n_obs_steps}")
        
        # 1. 加载 4 个 RGB backbone
        print("[DP2DP3] Loading Vision Backbones (Multi-GPU)...")
        # 支持通过 usr_args.gpu_ids 显式指定GPU (例如 --gpu_ids [0,1])
        requested_gpu_ids = self.usr_args.get("gpu_ids", None)
        if isinstance(requested_gpu_ids, (list, tuple)) and len(requested_gpu_ids) > 0:
            gpu_ids = list(requested_gpu_ids)
        else:
            # 优先把 backbone 放在用户指定的 GPU 上；如果有第二张卡，再把大模型放第二张卡
            if torch.cuda.device_count() > 1:
                other = 1 - self.gpu_id if self.gpu_id in (0, 1) else (self.gpu_id + 1) % torch.cuda.device_count()
                gpu_ids = [self.gpu_id, other]
            else:
                gpu_ids = [self.gpu_id]
        
        self.feature_extractors = MultiGPUFeatureExtractors(gpu_ids=gpu_ids)
        
        # 2. 加载对齐编码器 (RGB2PC)
        print("[DP2DP3] Loading Alignment Encoder...")
        # 假设对齐编码器路径在 config 中，或者我们使用 checkpoint 中保存的 config
        # 注意：训练脚本里 encoder path 是 config['encoder']['checkpoint']
        # 但是我们这里是加载 DP policy 的 checkpoint，它应该包含对齐编码器的信息吗？
        # train_online_from_config.py 加载了 encoder 并用来训练 policy
        # 但是 save_checkpoint 只保存了 policy state_dict?
        # 不，train_online_from_config.py 把 encoder 视为冻结的特征提取器，是否保存了它？
        # 看代码: torch.save({'policy': policy.state_dict(), ...})
        # 并没有保存 encoder！所以我们需要重新利用 config 中的路径加载 encoder。
        
        encoder_ckpt_path = config.get('encoder', {}).get('checkpoint')
        if not encoder_ckpt_path:
            # Fallback for offline checkpoints that missed this config
            encoder_ckpt_path = "/home/gl/RoboTwin/policy/DP2DP3/features_model/outputs/train_rgb2pc_runs/run_best_bs32/ckpt_step_0010000.pt"
            print(f"[DP2DP3] Warning: 'encoder.checkpoint' missing in config. Using default: {encoder_ckpt_path}")


        # 兼容：checkpoint 里可能保存的是相对路径（相对于 features_model 目录）
        encoder_ckpt_path = str(encoder_ckpt_path)
        if not os.path.isabs(encoder_ckpt_path):
            candidate = Path(features_model_dir) / encoder_ckpt_path
            if candidate.exists():
                encoder_ckpt_path = str(candidate)
        # 兜底：如果仍不存在，直接报出更清晰的信息
        if not Path(encoder_ckpt_path).exists():
            raise FileNotFoundError(
                f"Alignment encoder checkpoint not found: {encoder_ckpt_path} (raw={config.get('encoder', {}).get('checkpoint')})"
            )

        # ✅ 对齐训练/推理一致性校验：推理只提供每帧全局特征
        # 若对齐训练使用 token 池化或 window 粒度，会与推理输入分布不一致，导致成功率极低
        enc_ckpt = torch.load(encoder_ckpt_path, map_location="cpu")
        enc_args = enc_ckpt.get("args", {}) if isinstance(enc_ckpt, dict) else {}
        if not isinstance(enc_args, dict):
            enc_args = vars(enc_args)
        student_pool = str(enc_args.get("student_pool", "tokens"))
        sample_unit = str(enc_args.get("sample_unit", "step"))
        self.student_tokens = int(enc_args.get("student_tokens", 64))
        self.use_token_infer = student_pool == "tokens"
        if sample_unit != "step":
            raise ValueError(
                "对齐ckpt训练方式与推理不一致："
                f"student_pool={student_pool}, sample_unit={sample_unit}. "
                "当前推理仅支持 sample_unit=step。"
            )
        if self.use_token_infer:
            print(
                f"[DP2DP3] Token-level inference enabled. student_tokens={self.student_tokens}"
            )
        else:
            print("[DP2DP3] Using pooled-vector inference (student_pool=mean)")
            
        self.rgb_encoder = RGB2PCAlignedEncoder4Models.from_checkpoint(
            encoder_ckpt_path,
            map_location='cpu',
            freeze=True
        )
        self.rgb_encoder = self.rgb_encoder.to(self.device).eval()
        print(f"[DP2DP3] Alignment Encoder loaded from {encoder_ckpt_path}")

        # 对齐绕过模式（用于对比实验）
        self.align_dummy_mode = str(
            self.usr_args.get("align_dummy_mode", config.get("encoder", {}).get("dummy_mode", "none"))
        ).lower()
        if self.align_dummy_mode not in {"none", "pre_proj", "slice_mean"}:
            print(f"[DP2DP3] Unknown align_dummy_mode={self.align_dummy_mode}, fallback to 'none'")
            self.align_dummy_mode = "none"
        if self.align_dummy_mode != "none":
            print(f"[DP2DP3] Alignment dummy mode enabled: {self.align_dummy_mode}")

        # 3. 加载 Diffusion Policy Head
        print("[DP2DP3] Loading Policy Head...")
        
        # 判断 policy 类型
        # config['policy']['type'] == 'OfficialDP'
        policy_type = config.get('policy', {}).get('type', 'OfficialDP')
        if 'policy_type' in ckpt and ckpt.get('policy_type') == 'dual_stream':
            policy_type = 'DualStreamDP'
        # ✅ 新增：检测 DP-Aligned Policy（checkpoint 中记录了 policy_class）
        ckpt_policy_class = ckpt.get('policy_class', '')
        ckpt_policy_type = ckpt.get('policy_type', '')
        if ckpt_policy_class == 'DPAlignedPolicy' or ckpt_policy_type == 'dp_aligned':
            policy_type = 'DPAligned'
        elif ckpt_policy_class == 'DPAlignedDualStreamPolicy' or ckpt_policy_type == 'dp_aligned_dual_stream':
            policy_type = 'DPAlignedDualStream'
        use_proprio = bool(config.get('policy', {}).get('use_proprio', False))
        proprio_dim = int(config.get('policy', {}).get('proprio_dim', 14))
        proprio_mode = str(config.get('policy', {}).get('proprio_mode', 'concat'))
        proprio_hidden = int(config.get('policy', {}).get('proprio_hidden', 256))

        if policy_type == 'OfficialDP':
            # 这里的 obs_dim 是对齐后的 dim，即 fuse_dim * n_obs_steps
            # RGB2PCAlignedEncoder4Models 默认 fuse_dim=1280
            obs_dim = self.n_obs_steps * 1280
            
            self.policy = DPRGBPolicy(
                obs_dim=obs_dim,
                action_dim=self.model_action_dim,
                horizon=self.horizon,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.horizon,
                num_inference_steps=config.get('policy', {}).get('num_inference_steps', 100),
                use_proprio=use_proprio,
                proprio_dim=proprio_dim,
                proprio_mode=proprio_mode,
                proprio_hidden=proprio_hidden,
            )
        elif policy_type == 'DualStreamDP':
            obs_dim = self.n_obs_steps * 1280
            token_dim = int(ckpt.get('token_dim', 1280))
            self.policy = DPRGBDualStreamPolicy(
                obs_dim=obs_dim,
                token_dim=token_dim,
                action_dim=self.model_action_dim,
                horizon=self.horizon,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.horizon,
                num_inference_steps=config.get('policy', {}).get('num_inference_steps', 100),
                token_dropout=float(config.get('policy', {}).get('token_dropout', 0.0)),
                ctx_dropout=float(config.get('policy', {}).get('ctx_dropout', 0.0)),
                token_gate_init=float(config.get('policy', {}).get('token_gate_init', -4.0)),
                use_proprio=use_proprio,
                proprio_dim=proprio_dim,
                proprio_mode=proprio_mode,
                proprio_hidden=proprio_hidden,
            )
        elif policy_type == 'DPAligned':
            # DP-Aligned: agent_pos 直接 concat（无独立 MLP）
            # 从 checkpoint 或 config 读取 vis_projector 配置
            vis_projector_type = str(ckpt.get('vis_projector_type', config.get('policy', {}).get('vis_projector_type', 'none')))
            vis_projector_dim = int(ckpt.get('vis_projector_dim', config.get('policy', {}).get('vis_projector_dim', 1280)))
            self.policy = DPAlignedPolicy(
                vis_dim=1280,
                action_dim=self.model_action_dim,
                horizon=self.horizon,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.horizon,
                num_inference_steps=config.get('policy', {}).get('num_inference_steps', 100),
                use_proprio=use_proprio,
                proprio_dim=proprio_dim,
                vis_projector_type=vis_projector_type,
                vis_projector_dim=vis_projector_dim,
            )
        elif policy_type == 'DPAlignedDualStream':
            token_dim = int(ckpt.get('token_dim', 1280))
            self.policy = DPAlignedDualStreamPolicy(
                vis_dim=1280,
                token_dim=token_dim,
                action_dim=self.model_action_dim,
                horizon=self.horizon,
                n_obs_steps=self.n_obs_steps,
                n_action_steps=self.horizon,
                num_inference_steps=config.get('policy', {}).get('num_inference_steps', 100),
                token_dropout=float(config.get('policy', {}).get('token_dropout', 0.0)),
                ctx_dropout=float(config.get('policy', {}).get('ctx_dropout', 0.0)),
                token_gate_init=float(config.get('policy', {}).get('token_gate_init', -4.0)),
                use_proprio=use_proprio,
                proprio_dim=proprio_dim,
            )
        elif policy_type in ('SimpleMLP', 'SimpleDPHead'):
            # 与训练脚本 train_online_from_config.py 中的 SimpleDPHead 保持一致
            hidden_dim = int(config.get('policy', {}).get('hidden_dim', 512))
            obs_dim = self.n_obs_steps * 1280

            class SimpleDPHead(nn.Module):
                def __init__(self, obs_dim: int, action_dim: int, horizon: int, hidden_dim: int = 512):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(obs_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, action_dim * horizon),
                    )
                    self.action_dim = action_dim
                    self.horizon = horizon

                def forward(self, obs: torch.Tensor) -> torch.Tensor:
                    # obs: [B, To, D]
                    B = obs.shape[0]
                    obs_flat = obs.reshape(B, -1)
                    out = self.net(obs_flat)
                    return out.reshape(B, self.horizon, self.action_dim)

            self.policy = SimpleDPHead(
                obs_dim=obs_dim,
                action_dim=self.model_action_dim,
                horizon=self.horizon,
                hidden_dim=hidden_dim,
            )
        else:
            raise NotImplementedError(f"Policy type {policy_type} not supported in deployment yet")
            
        missing_keys, unexpected_keys = self.policy.load_state_dict(ckpt['policy'], strict=False)
        if missing_keys:
            print(f"[DP2DP3] Missing keys when loading policy: {missing_keys}")
        if unexpected_keys:
            print(f"[DP2DP3] Unexpected keys when loading policy: {unexpected_keys}")

        # ✅ 如果checkpoint中包含normalizer，则启用并加载
        if hasattr(self.policy, 'normalizer') and 'normalizer' in ckpt:
            self.policy.normalizer.load_state_dict(ckpt['normalizer'])
            self.policy.use_normalizer = True
            try:
                self.policy.normalizer.to(self.device)
            except Exception:
                pass

        # ✅ 若checkpoint仍处于gate warmup阶段，推理期强制gate与训练一致
        if isinstance(self.policy, (DPRGBDualStreamPolicy, DPAlignedDualStreamPolicy)):
            force_gate = self.usr_args.get("force_gate", None)
            force_gate_specified = "force_gate" in self.usr_args
            if force_gate is None:
                if not force_gate_specified:
                    ckpt_epoch = int(ckpt.get('epoch', 0))
                    train_cfg = config.get('train', {}) if isinstance(config, dict) else {}
                    total_epochs = int(train_cfg.get('epochs', 0)) if isinstance(train_cfg, dict) else 0
                    gate_warmup_epochs = int(
                        train_cfg.get('gate_warmup_epochs', total_epochs // 3 if total_epochs > 0 else 0)
                    )
                    if gate_warmup_epochs > 0 and ckpt_epoch > 0 and ckpt_epoch <= gate_warmup_epochs:
                        force_gate = 0.5
            if force_gate is not None:
                self.policy.force_gate = float(force_gate)
                print(
                    f"[DP2DP3] Force gate enabled: {self.policy.force_gate} "
                    f"(ckpt_epoch={ckpt.get('epoch', 'NA')})"
                )
        
        self.policy = self.policy.to(self.device).eval()
        
        self.model_loaded = True
        print("[DP2DP3] All models loaded successfully!")
        
    def reset(self):
        """重置观测缓冲区和动作队列"""
        try:
            self._episode_id = int(getattr(self, "_episode_id", -1)) + 1
        except Exception:
            self._episode_id = 0
        self.obs_buffer.clear()
        self.action_queue.clear()
        self._prev_action = None
        try:
            if hasattr(self, "_exec_action_hist") and self._exec_action_hist is not None:
                self._exec_action_hist.clear()
        except Exception:
            pass
        if hasattr(self, "feature_cache") and self.feature_cache is not None:
            try:
                self.feature_cache.clear()
            except Exception:
                pass
        self._action_log_current_plan_id = None
        self._append_action_log(
            {
                "stage": "episode_reset",
                "episode": int(self._episode_id),
                "exec_step": int(getattr(self, "_action_log_step_id", 0)),
            }
        )

    def _encode_pre_proj(self, features: torch.Tensor) -> torch.Tensor:
        """对齐绕过：返回融合后、proj_student之前的特征。"""
        b, t, m, _ = features.shape
        weight_dtype = next(self.rgb_encoder.parameters()).dtype
        feats = features.to(dtype=weight_dtype)
        zs = []
        for mi in range(m):
            ci = int(self.rgb_encoder.spec.in_dims[mi])
            tok = feats[:, :, mi, :ci].reshape(b * t, ci)
            z = self.rgb_encoder.adapters[mi](tok)
            zs.append(z)

        fused, _ = self.rgb_encoder.fusion(zs)
        fused = fused.reshape(b, t, -1)

        if self.rgb_encoder.use_context and self.rgb_encoder.context_encoder is not None:
            fused = self.rgb_encoder.pos_encoder(fused.transpose(0, 1)).transpose(0, 1)
            fused = self.rgb_encoder.context_encoder(fused)

        return fused

    @staticmethod
    def _encode_slice_mean(features: torch.Tensor) -> torch.Tensor:
        """对齐绕过：对每模型取前1280维并均值融合。"""
        fused = features[..., :1280].mean(dim=2)
        return fused
        
    def get_action(self, obs):
        """
        根据观测获取动作（兼容 RoBoTwin 框架）
        
        Args:
            obs: 编码后的观测字典
                - 'agent_pos': 关节状态
                - 'head_cam': RGB 图像 float [3, H, W]
                
        Returns:
            actions: 动作序列（用于 receding horizon）
        """
        # 将当前观测添加到缓冲区
        self.obs_buffer.append(obs)

        # 如果缓冲区还没满，用第一帧填充
        while len(self.obs_buffer) < self.n_obs_steps:
            self.obs_buffer.append(self.obs_buffer[0])

        # 🔧 可选：episode 开始时用当前关节状态初始化 prev_action，使第 1 步也受 rate limiter 约束
        if self.init_prev_action_from_obs and self._prev_action is None:
            try:
                ap = np.asarray(obs.get("agent_pos", []), dtype=np.float32).reshape(-1)
                if ap.shape[0] == int(self.env_action_dim):
                    self._prev_action = ap.copy()
                    if self.action_log_path:
                        self._append_action_log(
                            {
                                "stage": "init_prev_action",
                                "episode": int(getattr(self, "_episode_id", -1)),
                                "exec_step": int(getattr(self, "_action_log_step_id", 0)),
                            }
                        )
            except Exception:
                pass
        
        # 如果动作队列为空，需要预测新的动作序列
        if self.replan_every_call or len(self.action_queue) == 0:
            prev_queue_actions = None
            if (self.boundary_clamp or self.overlap_blend or self.plan_select_n > 1) and len(self.action_queue) > 0:
                try:
                    prev_queue_actions = np.stack(list(self.action_queue), axis=0).astype(np.float32)
                except Exception:
                    prev_queue_actions = None
            # 1. 准备图像 Batch
            # obs_buffer 里的图像是 np array [3, H, W] in [0, 1]
            images = []
            for o in self.obs_buffer:
                # 🔧 修复：确保与训练时完全一致的预处理
                img_np = (o['head_cam'].transpose(1, 2, 0) * 255.0).astype(np.uint8)
                # 确保值域在 [0, 255]
                img_np = np.clip(img_np, 0, 255)
                img_pil = Image.fromarray(img_np, mode='RGB')
                images.append(img_pil)

            # 1.5 准备本体感知（agent_pos）序列（若启用）
            agent_pos_tensor = None
            if bool(getattr(self.policy, "use_proprio", False)):
                try:
                    ap_list = []
                    for o in self.obs_buffer:
                        ap = np.asarray(o.get("agent_pos", []), dtype=np.float32).reshape(-1)
                        ap_list.append(ap)
                    if len(ap_list) > 0 and all(a.shape == ap_list[0].shape for a in ap_list):
                        ap_seq = np.stack(ap_list, axis=0)  # [To, D]
                        agent_pos_tensor = torch.from_numpy(ap_seq).float().to(self.device).unsqueeze(0)
                except Exception:
                    agent_pos_tensor = None
            
            # 2. 批量提取 RGB 特征
            # token-level: List[Tensor]，每个模型 [To, K, C_i]
            # pooled-level: Tensor [To, 4, 2048]
            obs_tokens = None
            if self.use_token_infer:
                if self.align_dummy_mode != "none":
                    print(
                        f"[DP2DP3] align_dummy_mode={self.align_dummy_mode} only supports pooled inference; "
                        "fallback to normal token encoder."
                    )
                max_tokens = None if self.token_sample_mode == "deterministic" else self.student_tokens
                if self.token_sample_seed is not None:
                    with torch.random.fork_rng(devices=[], enabled=True):
                        torch.manual_seed(int(self.token_sample_seed))
                        tokens_list = self.feature_extractors.extract_batch_tokens(
                            images,
                            max_tokens=max_tokens,
                            return_torch=True,
                        )
                else:
                    tokens_list = self.feature_extractors.extract_batch_tokens(
                        images,
                        max_tokens=max_tokens,
                        return_torch=True,
                    )
                if self.token_sample_mode == "deterministic":
                    fixed_tokens_list = []
                    for toks in tokens_list:
                        # toks: [To, K, C]
                        k = int(toks.shape[1])
                        target_k = int(self.student_tokens)
                        if k == target_k:
                            fixed_tokens_list.append(toks)
                            continue
                        if k > target_k:
                            idx = torch.linspace(0, k - 1, steps=target_k, device=toks.device)
                            idx = idx.round().long()
                            fixed_tokens_list.append(toks.index_select(1, idx))
                        else:
                            pad_idx = torch.linspace(0, k - 1, steps=target_k, device=toks.device)
                            pad_idx = pad_idx.round().long().clamp(0, k - 1)
                            fixed_tokens_list.append(toks.index_select(1, pad_idx))
                    tokens_list = fixed_tokens_list
                tokens_list = [t.to(self.device) for t in tokens_list]
                # [1, To, K, C_i] for each model
                tokens_list = [t.unsqueeze(0) for t in tokens_list]
                with torch.no_grad():
                    if isinstance(self.policy, (DPRGBDualStreamPolicy, DPAlignedDualStreamPolicy)):
                        obs_emb, obs_tokens = self.rgb_encoder(tokens_list, return_tokens=True)
                    else:
                        obs_emb = self.rgb_encoder(tokens_list)
            else:
                features_np = self.feature_extractors.extract_batch(images)
                features = torch.from_numpy(features_np).float().to(self.device).unsqueeze(0) # [1, To, 4, 2048]
                with torch.no_grad():
                    if self.align_dummy_mode == "pre_proj":
                        obs_emb = self._encode_pre_proj(features)
                    elif self.align_dummy_mode == "slice_mean":
                        obs_emb = self._encode_slice_mean(features)
                    else:
                        if isinstance(self.policy, (DPRGBDualStreamPolicy, DPAlignedDualStreamPolicy)):
                            obs_emb, obs_tokens = self.rgb_encoder(features, return_tokens=True)
                        else:
                            obs_emb = self.rgb_encoder(features)
            
            # 4. 通过 Diffusion Head 预测动作序列 [Ta, A]
            # 可选：Best-of-N 采样并选择最连续的 plan，抑制“回跳/反复来回”
            if int(self.plan_select_n) > 1:
                # overlap 参考：上一轮剩余动作（如果存在）
                overlap_ref = None
                if prev_queue_actions is not None and prev_queue_actions.ndim == 2 and prev_queue_actions.shape[0] > 0:
                    overlap_ref = prev_queue_actions

                # 评分需要当前关节状态（用于惩罚第1步的大跳变）
                try:
                    cur_qpos = np.asarray(obs.get("agent_pos", []), dtype=np.float32).reshape(-1)
                    if cur_qpos.shape[0] != int(self.env_action_dim):
                        cur_qpos = None
                except Exception:
                    cur_qpos = None

                # 为了可复现：用 seed_base + episode_id + plan_id 生成 base_seed
                ep = int(getattr(self, "_episode_id", -1))
                plan_id = int(getattr(self, "_action_log_plan_id", 0))
                base_seed = int(self.plan_select_seed_base) ^ (ep * 1000003) ^ (plan_id * 9176)

                mode = str(getattr(self, "plan_select_mode", "guard")).strip().lower()
                if mode not in ("guard", "min_cost"):
                    mode = "guard"

                # helper: compute overlap metrics
                def _overlap_metrics(cand: np.ndarray) -> tuple[float, float, int]:
                    if overlap_ref is None:
                        return 0.0, 0.0, 0
                    # 只比较“本次将执行的前 K 步”，避免用未来未执行部分过度约束策略
                    overlap = int(min(overlap_ref.shape[0], cand.shape[0], int(self.n_action_exec)))
                    if int(self.plan_select_overlap_steps) > 0:
                        overlap = min(overlap, int(self.plan_select_overlap_steps))
                    if overlap <= 0:
                        return 0.0, 0.0, 0
                    diff = cand[:overlap] - overlap_ref[:overlap]
                    mse = float(np.mean(diff * diff))
                    maxabs = float(np.max(np.abs(diff))) if diff.size else 0.0
                    return mse, maxabs, overlap

                # helper: detect “rewind risk” in action space
                # 若 cand[0] 非常接近过去(窗口内)某个已执行动作，则很可能出现“回到几秒前姿态重来”
                def _rewind_metrics(a0: np.ndarray) -> tuple[float, int, int]:
                    if not getattr(self, "rewind_guard", True):
                        return float("inf"), -1, 0
                    try:
                        hist = list(getattr(self, "_exec_action_hist", []))
                        t = int(len(hist))
                        wmin = int(getattr(self, "rewind_guard_window_min", 10))
                        wmax = int(getattr(self, "rewind_guard_window_max", 80))
                        if wmin <= 0 or wmax <= wmin or t <= wmin:
                            return float("inf"), -1, 0
                        start = max(0, t - wmax)
                        end = max(0, t - wmin)
                        if end <= start:
                            return float("inf"), -1, 0
                        past = np.stack(hist[start:end], axis=0).astype(np.float32, copy=False)  # [M,A]
                        a0v = np.asarray(a0, dtype=np.float32).reshape(1, -1)
                        if past.ndim != 2 or a0v.shape[1] != past.shape[1]:
                            return float("inf"), -1, int(past.shape[0]) if past.ndim == 2 else 0
                        diff = past - a0v  # [M,A]
                        mean_abs = np.mean(np.abs(diff), axis=1)  # [M]
                        j = int(mean_abs.argmin()) if mean_abs.size else -1
                        return float(mean_abs[j]) if j >= 0 else float("inf"), int(start + j) if j >= 0 else -1, int(past.shape[0])
                    except Exception:
                        return float("inf"), -1, 0

                n = int(self.plan_select_n)
                cand_costs = []
                cand_map: dict[int, np.ndarray] = {}

                # 候选0：不设置随机种子，直接走“当前 RNG 状态”的采样（最接近 baseline 行为）
                # 其它候选：用固定 seed 做 fork_rng 采样，用于在必要时挑选更连续的 plan
                cand0 = self._predict_action_sequence(obs_emb, obs_tokens, agent_pos_tensor, seed=None)
                cand_map[0] = cand0
                ov_mse0, ov_maxabs0, ov_k0 = _overlap_metrics(cand0)
                rw0, rw0_match_step, rw0_window = _rewind_metrics(cand0[0] if cand0.shape[0] > 0 else cand0)

                # cost_first: 第1步与当前关节状态的偏差（仅用于日志/可选min_cost）
                cost_first0 = 0.0
                if cur_qpos is not None and cand0.shape[0] > 0:
                    d0 = cand0[0] - cur_qpos
                    cost_first0 = float(np.mean(d0 * d0))
                # cost_smooth: plan 内部的平滑性（仅用于日志/可选min_cost）
                cost_smooth0 = 0.0
                if cand0.shape[0] >= 2:
                    dv = np.diff(cand0, axis=0)
                    cost_smooth0 = float(np.mean(dv * dv))
                # cost_cut: 关注 cut 点(K-1 -> K) 的大跳变。K=n_action_exec。
                # 该跳变会在“下一次重规划边界”显性表现为大跳/回跳，因此需要强惩罚。
                cost_cut_mse0 = 0.0
                cost_cut_maxabs0 = 0.0
                cut_k0 = 0
                try:
                    cut_k = int(self.n_action_exec)
                    if cut_k > 0 and cand0.shape[0] > cut_k:
                        dcut = cand0[cut_k] - cand0[cut_k - 1]
                        cost_cut_mse0 = float(np.mean(dcut * dcut))
                        cost_cut_maxabs0 = float(np.max(np.abs(dcut))) if dcut.size else 0.0
                        cut_k0 = int(cut_k)
                except Exception:
                    pass
                total0 = (
                    float(self.plan_select_w_overlap) * float(ov_mse0)
                    + float(self.plan_select_w_first) * float(cost_first0)
                    + float(self.plan_select_w_cut) * float(cost_cut_maxabs0 * cost_cut_maxabs0)
                    + float(self.plan_select_w_smooth) * float(cost_smooth0)
                )
                cand_costs.append(
                    {
                        "i": 0,
                        "seed": None,
                        "total": float(total0),
                        "overlap": float(ov_mse0),
                        "overlap_maxabs": float(ov_maxabs0),
                        "overlap_k": int(ov_k0),
                        "rewind_meanabs_min": float(rw0),
                        "rewind_match_step": int(rw0_match_step),
                        "rewind_window_len": int(rw0_window),
                        "first": float(cost_first0),
                        "cut_maxabs": float(cost_cut_maxabs0),
                        "cut_mse": float(cost_cut_mse0),
                        "cut_k": int(cut_k0),
                        "smooth": float(cost_smooth0),
                    }
                )

                best = cand0
                best_seed = None
                best_i = 0
                best_total = float(total0)
                best_overlap = float(ov_mse0)
                best_overlap_maxabs = float(ov_maxabs0)
                best_cut_maxabs = float(cost_cut_maxabs0)
                early_exit = False
                rewind_guard_triggered = False

                if mode == "guard":
                    need_search = False
                    try:
                        cut_thr = float(getattr(self, "plan_select_guard_cut_maxabs", 0.8))
                    except Exception:
                        cut_thr = 0.8
                    if overlap_ref is not None and int(ov_k0) > 0:
                        if (
                            float(ov_mse0) > float(self.plan_select_guard_overlap_mse)
                            or float(ov_maxabs0) > float(self.plan_select_guard_overlap_maxabs)
                            or float(cost_cut_maxabs0) > float(cut_thr)
                        ):
                            need_search = True
                    else:
                        # episode 起始：即使没有 overlap_ref，也要防止 plan 内部在 cut 点出现超大跳变，
                        # 否则会在下一次重规划边界形成“看起来瞬移/回跳”的大跳。
                        if float(cost_cut_maxabs0) > float(cut_thr):
                            need_search = True
                    # rewind guard：若第1步非常接近历史动作，则强制搜索其它候选
                    try:
                        thr = float(getattr(self, "rewind_guard_thresh", 0.05))
                    except Exception:
                        thr = 0.05
                    if np.isfinite(float(rw0)) and float(rw0) < float(thr):
                        need_search = True
                        rewind_guard_triggered = True
                    if not need_search:
                        early_exit = True
                    else:
                        # 只有在候选0 与上一段剩余动作差异过大时，才采样其它候选并选最“连续”的
                        for i in range(1, n):
                            seed_i = int(base_seed + (i - 1))
                            cand = self._predict_action_sequence(obs_emb, obs_tokens, agent_pos_tensor, seed=seed_i)
                            cand_map[int(i)] = cand
                            ov_mse, ov_maxabs, ov_k = _overlap_metrics(cand)
                            rw, rw_match_step, rw_window = _rewind_metrics(cand[0] if cand.shape[0] > 0 else cand)

                            cost_first = 0.0
                            if cur_qpos is not None and cand.shape[0] > 0:
                                d0 = cand[0] - cur_qpos
                                cost_first = float(np.mean(d0 * d0))
                            cost_smooth = 0.0
                            if cand.shape[0] >= 2:
                                dv = np.diff(cand, axis=0)
                                cost_smooth = float(np.mean(dv * dv))
                            cost_cut_mse = 0.0
                            cost_cut_maxabs = 0.0
                            cut_k = 0
                            try:
                                k = int(self.n_action_exec)
                                if k > 0 and cand.shape[0] > k:
                                    dcut = cand[k] - cand[k - 1]
                                    cost_cut_mse = float(np.mean(dcut * dcut))
                                    cost_cut_maxabs = float(np.max(np.abs(dcut))) if dcut.size else 0.0
                                    cut_k = int(k)
                            except Exception:
                                pass
                            total = (
                                float(self.plan_select_w_overlap) * float(ov_mse)
                                + float(self.plan_select_w_first) * float(cost_first)
                                + float(self.plan_select_w_cut) * float(cost_cut_maxabs * cost_cut_maxabs)
                                + float(self.plan_select_w_smooth) * float(cost_smooth)
                            )
                            cand_costs.append(
                                {
                                    "i": int(i),
                                    "seed": int(seed_i),
                                    "total": float(total),
                                    "overlap": float(ov_mse),
                                    "overlap_maxabs": float(ov_maxabs),
                                    "overlap_k": int(ov_k),
                                    "rewind_meanabs_min": float(rw),
                                    "rewind_match_step": int(rw_match_step),
                                    "rewind_window_len": int(rw_window),
                                    "first": float(cost_first),
                                    "cut_maxabs": float(cost_cut_maxabs),
                                    "cut_mse": float(cost_cut_mse),
                                    "cut_k": int(cut_k),
                                    "smooth": float(cost_smooth),
                                }
                            )

                        # 选择：直接按 total 最小（total 已包含 overlap + first + cut + smooth）
                        # cut 惩罚能显著减少“每 4 步重规划边界”的大跳/回跳。
                        best_c = min(
                            cand_costs,
                            key=lambda c: (
                                float(c.get("total", 0.0)),
                                float(c.get("overlap_maxabs", 0.0)),
                                float(c.get("cut_maxabs", 0.0)),
                            ),
                        )
                        best_i = int(best_c.get("i", 0))
                        best_seed = best_c.get("seed", None)
                        best = cand_map.get(int(best_i), cand0)
                        # recompute best totals for logging
                        best_total = float(best_c.get("total", best_total))
                        best_overlap = float(best_c.get("overlap", best_overlap))
                        best_overlap_maxabs = float(best_c.get("overlap_maxabs", best_overlap_maxabs))
                        best_cut_maxabs = float(best_c.get("cut_maxabs", best_cut_maxabs))
                else:
                    # min_cost：全量采样 N 个候选并按加权总 cost 选择（更激进，可能更慢/更保守）
                    for i in range(1, n):
                        seed_i = int(base_seed + (i - 1))
                        cand = self._predict_action_sequence(obs_emb, obs_tokens, agent_pos_tensor, seed=seed_i)
                        cand_map[int(i)] = cand
                        ov_mse, ov_maxabs, ov_k = _overlap_metrics(cand)
                        rw, rw_match_step, rw_window = _rewind_metrics(cand[0] if cand.shape[0] > 0 else cand)

                        cost_first = 0.0
                        if cur_qpos is not None and cand.shape[0] > 0:
                            d0 = cand[0] - cur_qpos
                            cost_first = float(np.mean(d0 * d0))
                        cost_smooth = 0.0
                        if cand.shape[0] >= 2:
                            dv = np.diff(cand, axis=0)
                            cost_smooth = float(np.mean(dv * dv))
                        cost_cut_mse = 0.0
                        cost_cut_maxabs = 0.0
                        cut_k = 0
                        try:
                            k = int(self.n_action_exec)
                            if k > 0 and cand.shape[0] > k:
                                dcut = cand[k] - cand[k - 1]
                                cost_cut_mse = float(np.mean(dcut * dcut))
                                cost_cut_maxabs = float(np.max(np.abs(dcut))) if dcut.size else 0.0
                                cut_k = int(k)
                        except Exception:
                            pass
                        total = (
                            float(self.plan_select_w_overlap) * float(ov_mse)
                            + float(self.plan_select_w_first) * float(cost_first)
                            + float(self.plan_select_w_cut) * float(cost_cut_maxabs * cost_cut_maxabs)
                            + float(self.plan_select_w_smooth) * float(cost_smooth)
                        )
                        cand_costs.append(
                            {
                                "i": int(i),
                                "seed": int(seed_i),
                                "total": float(total),
                                "overlap": float(ov_mse),
                                "overlap_maxabs": float(ov_maxabs),
                                "overlap_k": int(ov_k),
                                "rewind_meanabs_min": float(rw),
                                "rewind_match_step": int(rw_match_step),
                                "rewind_window_len": int(rw_window),
                                "first": float(cost_first),
                                "cut_maxabs": float(cost_cut_maxabs),
                                "cut_mse": float(cost_cut_mse),
                                "cut_k": int(cut_k),
                                "smooth": float(cost_smooth),
                            }
                        )

                        if float(total) < float(best_total):
                            best = cand
                            best_seed = seed_i
                            best_i = int(i)
                            best_total = float(total)
                            best_overlap = float(ov_mse)
                            best_overlap_maxabs = float(ov_maxabs)
                            best_cut_maxabs = float(cost_cut_maxabs)

                action_pred = np.asarray(best, dtype=np.float32)

                # 极端不一致时的“尾巴回退”：若即使 Best-of-N 也无法与上一段 tail 对齐，
                # 则本次执行直接沿用上一段剩余动作（避免边界处突然“回到过去姿态”）。
                tail_fallback = False
                if mode == "guard" and overlap_ref is not None and int(best_i) >= 0:
                    try:
                        thr_mse = float(getattr(self, "plan_select_guard_overlap_mse", 0.05))
                        thr_maxabs = float(getattr(self, "plan_select_guard_overlap_maxabs", 0.5))
                    except Exception:
                        thr_mse, thr_maxabs = 0.05, 0.5
                    try:
                        overlap = int(min(overlap_ref.shape[0], action_pred.shape[0], int(self.n_action_exec)))
                        if int(self.plan_select_overlap_steps) > 0:
                            overlap = min(overlap, int(self.plan_select_overlap_steps))
                        if overlap > 0 and (float(best_overlap) > float(thr_mse)) and (float(best_overlap_maxabs) > float(thr_maxabs)):
                            tail_fallback = True
                            before = action_pred[:overlap].copy()
                            # 关键：不要把“旧 tail 的前缀”与“新候选的后缀”硬拼接。
                            # 这会在 prefix(第K步) -> suffix(第K+1步) 处制造一个新的巨大断点，
                            # 反而导致下一次重规划边界出现回跳。
                            # 更安全的做法：本次直接执行旧 tail（长度=overlap），并让队列在执行完后尽快重新规划。
                            action_pred = overlap_ref[:overlap].astype(np.float32, copy=True)
                            if self.action_log_path:
                                db = np.abs(before - overlap_ref[:overlap])
                                self._append_action_log(
                                    {
                                        "stage": "plan_select_tail_fallback",
                                        "episode": int(getattr(self, "_episode_id", -1)),
                                        "plan_id": int(self._action_log_plan_id),
                                        "exec_step": int(self._action_log_step_id),
                                        "overlap": int(overlap),
                                        "max_abs_before": float(db.max()) if db.size else 0.0,
                                        "best_overlap": float(best_overlap),
                                        "best_overlap_maxabs": float(best_overlap_maxabs),
                                    }
                                )
                    except Exception:
                        pass

                if self.action_log_path:
                    self._append_action_log(
                        {
                            "stage": "plan_select",
                            "episode": int(getattr(self, "_episode_id", -1)),
                            "plan_id": int(self._action_log_plan_id),
                            "exec_step": int(self._action_log_step_id),
                            "mode": str(mode),
                            "early_exit": bool(early_exit),
                            "rewind_guard_triggered": bool(rewind_guard_triggered),
                            "tail_fallback": bool(tail_fallback),
                            "guard_overlap_mse": float(getattr(self, "plan_select_guard_overlap_mse", 0.05)),
                            "guard_overlap_maxabs": float(getattr(self, "plan_select_guard_overlap_maxabs", 0.5)),
                            "guard_cut_maxabs": float(getattr(self, "plan_select_guard_cut_maxabs", 0.8)),
                            "n": int(self.plan_select_n),
                            "base_seed": int(base_seed),
                            "best_i": int(best_i),
                            "best_seed": (int(best_seed) if best_seed is not None else None),
                            "best_total": float(best_total),
                            "best_overlap": float(best_overlap),
                            "best_overlap_maxabs": float(best_overlap_maxabs),
                            "best_cut_maxabs": float(best_cut_maxabs),
                            "candidates": cand_costs,
                        }
                    )

                # guard 模式下：若新 plan 与上一段 tail 仍差异很大，用“逐步过渡”的 ramp-blend 缓解边界回跳
                # 说明：与全局 overlap_blend 不同，这里只在必要时启用，并采用从 ref->new 的线性权重递增，
                # 避免只裁剪第1步导致 plan 内部出现“0->1 步巨大跳变”。
                if mode == "guard" and overlap_ref is not None and bool(getattr(self, "plan_select_ramp_blend", False)):
                    try:
                        overlap = int(min(overlap_ref.shape[0], action_pred.shape[0], int(self.n_action_exec)))
                        if int(self.plan_select_overlap_steps) > 0:
                            overlap = min(overlap, int(self.plan_select_overlap_steps))
                        if overlap > 0:
                            mismatch = (
                                float(best_overlap) > float(getattr(self, "plan_select_guard_overlap_mse", 0.05))
                                or float(best_overlap_maxabs) > float(getattr(self, "plan_select_guard_overlap_maxabs", 0.5))
                            )
                            if mismatch:
                                ref = overlap_ref[:overlap].astype(np.float32, copy=False)
                                before = action_pred[:overlap].copy()
                                # ramp: w = (t+1)/overlap
                                for t in range(overlap):
                                    w = float(t + 1) / float(overlap)
                                    action_pred[t] = (1.0 - w) * ref[t] + w * before[t]
                                if self.action_log_path:
                                    db = np.abs(before - ref)
                                    da = np.abs(action_pred[:overlap] - ref)
                                    self._append_action_log(
                                        {
                                            "stage": "plan_select_ramp_blend",
                                            "episode": int(getattr(self, "_episode_id", -1)),
                                            "plan_id": int(self._action_log_plan_id),
                                            "exec_step": int(self._action_log_step_id),
                                            "overlap": int(overlap),
                                            "max_abs_before": float(db.max()) if db.size else 0.0,
                                            "max_abs_after": float(da.max()) if da.size else 0.0,
                                            "best_overlap": float(best_overlap),
                                            "best_overlap_maxabs": float(best_overlap_maxabs),
                                        }
                                    )
                    except Exception:
                        pass
            else:
                action_pred = self._predict_action_sequence(obs_emb, obs_tokens, agent_pos_tensor, seed=None).astype(np.float32, copy=False)
            
            # 🔍 调试输出1: 模型原始输出
            if self.debug_actions:
                print(f"[DEBUG-Aligned] 模型原始输出:")
                print(f"  Shape: {action_pred.shape}")
                print(f"  Range: [{action_pred.min():.3f}, {action_pred.max():.3f}]")
                print(f"  Mean: {action_pred.mean():.3f}, Std: {action_pred.std():.3f}")
                print(f"  First action: {action_pred[0]}")

            # 为了兼容旧逻辑：这里保留 clip 前副本（用于 debug）
            action_pred_before_clip = action_pred.copy()

            # 🔧 计划重叠融合（plan overlap blending）
            # 将新 plan 的前缀与“上一段剩余 plan”做融合，减少重规划导致的来回反复。
            if self.overlap_blend and prev_queue_actions is not None and prev_queue_actions.ndim == 2:
                try:
                    alpha = float(self.overlap_blend_alpha)
                    alpha = max(0.0, min(1.0, alpha))
                    overlap = min(prev_queue_actions.shape[0], action_pred.shape[0])
                    max_steps = int(self.overlap_blend_steps) if int(self.overlap_blend_steps) > 0 else overlap
                    overlap = min(overlap, max_steps)
                    if overlap > 0 and alpha < 1.0:
                        ref = prev_queue_actions[:overlap]
                        blended = (1.0 - alpha) * ref + alpha * action_pred[:overlap]
                        if self.action_log_path:
                            delta_before = np.abs(action_pred[:overlap] - ref)
                            delta_after = np.abs(blended - ref)
                            if not np.allclose(action_pred[:overlap], blended):
                                self._append_action_log(
                                    {
                                        "stage": "overlap_blend",
                                        "episode": int(getattr(self, "_episode_id", -1)),
                                        "plan_id": int(self._action_log_plan_id),
                                        "exec_step": int(self._action_log_step_id),
                                        "overlap": int(overlap),
                                        "alpha": float(alpha),
                                        "max_abs_before": float(delta_before.max()) if delta_before.size else 0.0,
                                        "max_abs_after": float(delta_after.max()) if delta_after.size else 0.0,
                                    }
                                )
                        action_pred[:overlap] = blended
                        action_pred = np.clip(action_pred, -3.0, 3.0)
                except Exception:
                    pass

            # 🔧 Chunk 边界一致性约束：仅对“上一轮剩余动作”对应的前缀做大跳变裁剪
            # 仅在启用 boundary_clamp 且存在上一轮剩余动作时生效。
            if self.boundary_clamp and prev_queue_actions is not None and prev_queue_actions.ndim == 2:
                overlap = min(prev_queue_actions.shape[0], action_pred.shape[0], int(self.boundary_clamp_steps))
                if overlap > 0:
                    ref = prev_queue_actions[:overlap]
                    diff = action_pred[:overlap] - ref
                    max_d = float(self.boundary_max_delta)
                    clipped = np.clip(diff, -max_d, max_d)
                    if self.action_log_path and not np.allclose(clipped, diff):
                        self._append_action_log({
                            "stage": "boundary_clamp",
                            "episode": int(getattr(self, "_episode_id", -1)),
                            "plan_id": int(self._action_log_plan_id),
                            "exec_step": int(self._action_log_step_id),
                            "overlap": int(overlap),
                            "max_delta": float(max_d),
                            "max_abs_before": float(np.abs(diff).max()),
                            "max_abs_after": float(np.abs(clipped).max()),
                            "num_clipped": int(np.sum(np.abs(diff) > max_d)),
                        })
                    action_pred[:overlap] = ref + clipped
                    action_pred = np.clip(action_pred, -3.0, 3.0)

            # 🔧 逐步变化率限制（rate limiter）：对 action_pred 做逐步Δ裁剪，避免“瞬移/闪现”
            if self.step_delta_clamp and self._prev_action is not None:
                try:
                    prev = np.asarray(self._prev_action, dtype=np.float32).reshape(-1)
                    max_d = float(self.step_max_delta)
                    if prev.shape[0] == int(self.env_action_dim) and max_d > 0:
                        seq = np.asarray(action_pred, dtype=np.float32).copy()
                        clamp_steps = int(self.step_delta_clamp_steps)
                        if clamp_steps <= 0:
                            clamp_steps = 0
                        clamp_steps = min(int(seq.shape[0]), clamp_steps) if clamp_steps else 0
                        diffs = []
                        clipped_diffs = []
                        changed = False
                        for t in range(clamp_steps):
                            d = seq[t] - prev
                            cd = np.clip(d, -max_d, max_d)
                            diffs.append(float(np.abs(d).max()))
                            clipped_diffs.append(float(np.abs(cd).max()))
                            if not np.allclose(cd, d):
                                changed = True
                            seq[t] = prev + cd
                            prev = seq[t]
                        if changed:
                            if self.action_log_path:
                                self._append_action_log({
                                    "stage": "step_delta_clamp",
                                    "episode": int(getattr(self, "_episode_id", -1)),
                                    "plan_id": int(self._action_log_plan_id),
                                    "exec_step": int(self._action_log_step_id),
                                    "max_delta": float(max_d),
                                    "max_abs_before": float(np.max(diffs) if diffs else 0.0),
                                    "max_abs_after": float(np.max(clipped_diffs) if clipped_diffs else 0.0),
                                })
                            seq = np.clip(seq, -3.0, 3.0)
                            action_pred = seq
                except Exception:
                    pass

            # 记录规划的动作序列
            plan_id = int(self._action_log_plan_id)
            self._action_log_current_plan_id = plan_id
            self._append_action_log({
                "stage": "plan",
                "episode": int(getattr(self, "_episode_id", -1)),
                "plan_id": plan_id,
                "exec_step": int(self._action_log_step_id),
                "actions": action_pred.tolist(),
            })
            self._action_log_plan_id += 1
            
            # 🔍 调试输出3: Clip检查
            if self.debug_actions:
                if not np.allclose(action_pred, action_pred_before_clip):
                    print(f"\n[WARNING-Aligned] 动作被Clip! 原始range: [{action_pred_before_clip.min():.3f}, {action_pred_before_clip.max():.3f}]")
            
            # 🔍 调试输出4: 动作变化率（检测是否卡住）
            if self.debug_actions:
                if len(self.action_queue) > 0:
                    last_action = list(self.action_queue)[-1]
                    action_diff = np.abs(action_pred[0] - last_action).mean()
                    print(f"\n[DEBUG-Aligned] 与上一动作的差异: {action_diff:.4f}")
                    if action_diff < 0.01:
                        print(f"  ⚠️  动作几乎不变！可能卡住")
            
            if self.debug_actions:
                print(f"\n[DEBUG-Aligned] 即将执行的动作序列 (前3步):")
                for i in range(min(3, len(action_pred))):
                    print(f"  Step {i}: {action_pred[i]}")

            # 部署期不再做 12->14 补零兜底（会导致夹爪不动）。必须严格一致。
            if int(action_pred.shape[-1]) != int(self.env_action_dim):
                raise RuntimeError(
                    f"Model action dim {action_pred.shape[-1]} does not match env expected {self.env_action_dim}. "
                    "请重新训练 14 维 head。"
                )

            # [已移除] 轻量低通滤波：因为每次推理生成整个Horizon的动作并Open-loop执行，
            # 跨Chunk的平滑会导致显著的延迟（Lag），且破坏了Diffusion生成轨迹的物理一致性。
            # 如果需要平滑，建议在 Receding Horizon (Predict H, Execute K, K<H) 的架构下做 Temporal Ensembling。
            #
            # if self.action_smoothing_alpha > 0:
            #     alpha = float(self.action_smoothing_alpha)
            #     alpha = max(0.0, min(1.0, alpha))
            #     smoothed = []
            #     prev = self._prev_action
            #     for a in action_pred:
            #         a = a.astype(np.float32)
            #         if prev is None:
            #             prev = a
            #         else:
            #             prev = (1.0 - alpha) * prev + alpha * a
            #         smoothed.append(prev)
            #     self._prev_action = prev
            #     action_pred = np.stack(smoothed, axis=0)
            
            # 将动作序列加入队列
            # Receding Horizon Control: 每次重新规划时清空旧队列，确保使用最新观测生成的动作
            self.action_queue.clear()
            self.action_queue.extend(action_pred)
        
        # 🔧 修复：Receding Horizon - 只返回前 n_action_exec 步
        # 标准 DP 策略：Predict H, Execute K (K < H). 通常 K=H/2 或固定值
        # 这里设置为 4 (配合 n_obs_steps=4, Horizon=8) 或者由参数决定
        # 为了保证实时性和平滑性，建议执行步数不要太长
        n_action_exec = min(self.n_action_exec, len(self.action_queue))
        return list(self.action_queue)[:n_action_exec]
    
    def update_obs(self, obs):
        """更新观测缓冲区（兼容 RoBoTwin 框架）"""
        self.obs_buffer.append(obs)
    
    def pop_action(self):
        """从动作队列中取出一个动作"""
        # Receding Horizon 逻辑需配合 eval 循环：
        # 如果我们在 eval 中循环执行了 get_action 返回的所有动作，
        # 那么队列会被 pop 空，下一次 get_action 就会重新规划。
        # 如果我们希望在队列还剩一部分为了平滑连接时就重规划，需要修改 get_action 的触发逻辑。
        # 当前修改为：每次 get_action 若队列空则规划。
        # 为了强制 Receding Horizon (Predict 8 Execute 4)，eval 端需要只执行 4 步。
        # 但标准接口 eval(model, obs) 是一步一步调用的。
        # 我们通过控制 get_action 返回的长度来控制 update_obs 的频率吗？
        # 不，eval 函数里是 for action in actions: execute -> update_obs -> pop.
        # 所以 get_action 返回几步，就会执行几步，然后再次调用 get_action.
        # 上面的修改 get_action 返回 4 步，意味着执行 4 步 (Update 4 次 obs) 后重新规划。
        # 这是正确的 Receding Horizon 逻辑 (Exec K=4).
        if len(self.action_queue) > 0:
            a = self.action_queue.popleft()
            try:
                self._prev_action = np.asarray(a, dtype=np.float32)
            except Exception:
                self._prev_action = None
            try:
                if hasattr(self, "_exec_action_hist") and self._exec_action_hist is not None:
                    self._exec_action_hist.append(np.asarray(a, dtype=np.float32).copy())
            except Exception:
                pass
            return a
        return np.zeros(self.action_dim, dtype=np.float32)


def get_model(usr_args):
    """
    RoBoTwin 标准接口：获取 policy 模型实例
    
    Args:
        usr_args: 配置参数，包含：
            - task_name: 任务名称
            - ckpt_setting: checkpoint 设置
            - expert_data_num: 专家数据数量
            - checkpoint_num: checkpoint 编号
            - seed: 随机种子
            - left_arm_dim: 左臂维度
            - right_arm_dim: 右臂维度
            
    Returns:
        DP2DP3Model 实例
    """
    model = DP2DP3Model(usr_args)
    return model


def _safe_get_tcp_pose(task_env):
    """
    尝试从环境中读取末端执行器（TCP）位姿，用于量化“闪现/回跳”。
    返回格式：
      {"left": [x,y,z,...] or None, "right": [x,y,z,...] or None}
    若无法读取则返回 None。
    """
    try:
        robot = getattr(task_env, "robot", None)
        if robot is None:
            return None
        out = {}
        if hasattr(robot, "get_left_tcp_pose"):
            try:
                out["left"] = [float(x) for x in list(robot.get_left_tcp_pose())]
            except Exception:
                out["left"] = None
        if hasattr(robot, "get_right_tcp_pose"):
            try:
                out["right"] = [float(x) for x in list(robot.get_right_tcp_pose())]
            except Exception:
                out["right"] = None
        return out if out else None
    except Exception:
        return None


def eval(TASK_ENV, model, observation):
    """
    RoBoTwin 标准接口：执行一步推理
    
    Args:
        TASK_ENV: 任务环境类，用于与环境交互
        model: get_model() 返回的模型实例
        observation: 环境观测
        
    🔧 修复: obs/action同步问题
    - 问题: 每执行一个action就更新一次obs，导致缓冲区全是新观测，丢失历史
    - 修复: 只在执行完所有动作后更新一次观测，保持时序合理性
    """
    obs = encode_obs(observation)
    
    # 获取动作序列
    actions = model.get_action(obs)
    
    # 🔧 修复: 执行每一步后更新观测，确保缓冲区始终是最新的n_obs_steps
    for i, action in enumerate(actions):
        # 记录执行前状态（用于衡量“瞬移”）
        before_joint = None
        try:
            before_joint = np.asarray(observation.get("joint_action", {}).get("vector", []), dtype=np.float32).reshape(-1).tolist()
        except Exception:
            before_joint = None
        before_tcp = _safe_get_tcp_pose(TASK_ENV)

        TASK_ENV.take_action(action)
        model.pop_action()

        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)

        # 记录执行后状态（动作日志）
        if hasattr(model, "_append_action_log"):
            after_joint = None
            try:
                after_joint = np.asarray(observation.get("joint_action", {}).get("vector", []), dtype=np.float32).reshape(-1).tolist()
            except Exception:
                after_joint = None
            after_tcp = _safe_get_tcp_pose(TASK_ENV)
            pid = getattr(model, "_action_log_current_plan_id", None)
            try:
                pid = int(pid) if pid is not None else -1
            except Exception:
                pid = -1
            model._append_action_log({
                "stage": "exec",
                "episode": int(getattr(model, "_episode_id", -1)),
                "step": int(getattr(model, "_action_log_step_id", 0)),
                "plan_id": pid,
                "action": np.asarray(action).tolist(),
                "agent_pos_before": before_joint,
                "agent_pos_after": after_joint,
                "tcp_before": before_tcp,
                "tcp_after": after_tcp,
            })
            if hasattr(model, "_action_log_step_id"):
                model._action_log_step_id += 1


def reset_model(model):
    """
    RoBoTwin 标准接口：重置模型状态
    
    Args:
        model: get_model() 返回的模型实例
    """
    model.reset()
