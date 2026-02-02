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

    def forward(self, obs):
        """推理模式"""
        B = obs.shape[0]
        device = obs.device
        
        # 编码观测
        obs_flat = obs.reshape(B, -1)
        obs_cond = self.obs_encoder(obs_flat)
        
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

    def _build_cond(self, obs_global: torch.Tensor, obs_tokens: torch.Tensor) -> torch.Tensor:
        b = obs_global.shape[0]
        obs_flat = obs_global.reshape(b, -1)
        global_cond = self.obs_encoder(obs_flat)
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

    def forward(self, obs_global: torch.Tensor, obs_tokens: torch.Tensor):
        b = obs_global.shape[0]
        device = obs_global.device
        obs_cond = self._build_cond(obs_global, obs_tokens)

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
        self.replan_every_call = True
        self.n_action_exec = int(usr_args.get('n_action_exec', 4))
        
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

        # 可选：固定token采样随机种子（减少随机token导致的动作抖动）
        self.token_sample_seed = usr_args.get('token_sample_seed', None)
        # token采样模式: random | deterministic
        self.token_sample_mode = str(usr_args.get('token_sample_mode', 'random')).lower()
        
        print(f"[DP2DP3] Model initialization complete")
        print(f"  Task: {self.task_name}")
        print(f"  Horizon: {self.horizon}, N_obs_steps: {self.n_obs_steps}")
        print(f"  Action dim: {self.action_dim}")

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
                num_inference_steps=config.get('policy', {}).get('num_inference_steps', 100)
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
        if isinstance(self.policy, DPRGBDualStreamPolicy):
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
        self.obs_buffer.clear()
        self.action_queue.clear()
        self._prev_action = None

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
        
        # 如果动作队列为空，需要预测新的动作序列
        if self.replan_every_call or len(self.action_queue) == 0:
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
                    if isinstance(self.policy, DPRGBDualStreamPolicy):
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
                        if isinstance(self.policy, DPRGBDualStreamPolicy):
                            obs_emb, obs_tokens = self.rgb_encoder(features, return_tokens=True)
                        else:
                            obs_emb = self.rgb_encoder(features)
            
            # 4. 通过 Diffusion Head 预测动作序列 [1, Ta, A]
            with torch.no_grad():
                if isinstance(self.policy, DPRGBDualStreamPolicy):
                    if obs_tokens is None:
                        raise RuntimeError("DualStream policy requires token features, but obs_tokens is None")
                    action_pred = self.policy(obs_emb, obs_tokens)
                else:
                    action_pred = self.policy(obs_emb)
            
            action_pred = action_pred.squeeze(0).cpu().numpy()  # [Ta, model_A]
            
            # 🔍 调试输出1: 模型原始输出（归一化后的值，应该在[-1,1]范围）
            print(f"[DEBUG-Aligned] 模型原始输出:")
            print(f"  Shape: {action_pred.shape}")
            print(f"  Range: [{action_pred.min():.3f}, {action_pred.max():.3f}]")
            print(f"  Mean: {action_pred.mean():.3f}, Std: {action_pred.std():.3f}")
            print(f"  First action: {action_pred[0]}")

            # 归一化反变换（若policy没有内置normalizer，才使用action_stats）
            action_pred_before_denorm = action_pred.copy()  # 保存用于调试
            if not getattr(self.policy, 'use_normalizer', False) and self.action_stats is not None:
                action_min = np.array(self.action_stats.get('min', -3.0), dtype=np.float32)
                action_max = np.array(self.action_stats.get('max', 3.0), dtype=np.float32)
                action_pred = (action_pred + 1.0) * 0.5 * (action_max - action_min) + action_min
                
                # 🔍 调试输出2: 反归一化后的值
                print(f"\n[DEBUG-Aligned] 反归一化后:")
                print(f"  Range: [{action_pred.min():.3f}, {action_pred.max():.3f}]")
                print(f"  Mean: {action_pred.mean():.3f}, Std: {action_pred.std():.3f}")
                print(f"  First action: {action_pred[0]}")
            
            # 🔧 重要：检查训练时是否做了归一化
            # 如果训练时做了归一化，推理时必须反归一化
            # 如果训练时没有归一化，推理时不需要反归一化
            
            # 🔧 安全限制：防止异常值（根据训练数据范围设置）
            action_pred_before_clip = action_pred.copy()
            action_pred = np.clip(action_pred, -3.0, 3.0)

            # 记录规划的动作序列
            self._append_action_log({
                "stage": "plan",
                "plan_id": int(self._action_log_plan_id),
                "actions": action_pred.tolist(),
            })
            self._action_log_plan_id += 1
            
            # 🔍 调试输出3: Clip检查
            if not np.allclose(action_pred, action_pred_before_clip):
                print(f"\n[WARNING-Aligned] 动作被Clip! 原始range: [{action_pred_before_clip.min():.3f}, {action_pred_before_clip.max():.3f}]")
            
            # 🔍 调试输出4: 动作变化率（检测是否卡住）
            if len(self.action_queue) > 0:
                last_action = list(self.action_queue)[-1]
                action_diff = np.abs(action_pred[0] - last_action).mean()
                print(f"\n[DEBUG-Aligned] 与上一动作的差异: {action_diff:.4f}")
                if action_diff < 0.01:
                    print(f"  ⚠️  动作几乎不变！可能卡住")
            
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
            return self.action_queue.popleft()
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
        TASK_ENV.take_action(action)
        model.pop_action()

        if hasattr(model, "_append_action_log"):
            model._append_action_log({
                "stage": "exec",
                "step": int(getattr(model, "_action_log_step_id", 0)),
                "action": np.asarray(action).tolist(),
            })
            if hasattr(model, "_action_log_step_id"):
                model._action_log_step_id += 1

        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    """
    RoBoTwin 标准接口：重置模型状态
    
    Args:
        model: get_model() 返回的模型实例
    """
    model.reset()

