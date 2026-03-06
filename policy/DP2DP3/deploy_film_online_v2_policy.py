"""DP2DP3 deploy_film_online_v2_policy.py

在线 2 模型 DA3-FiLM v2 部署脚本 (DINOv3 + DA3 + AttentionPooling)
====================================================================

与 v1 (deploy_film_online_policy.py) 的区别:
  - Encoder: DA3Film2ModelEncoderV2 (AttentionPooling 替代 mean_pool)
  - Checkpoint 目录: checkpoints_film_online_v2/
  - encoder_cfg 读取 max_sem_tokens / max_geo_tokens (而非 max_tokens)
  - 推理时 extract_batch_tokens 使用 max(max_sem_tokens, max_geo_tokens)

RoBoTwin 标准接口:
  get_model(usr_args) -> Film2ModelOnlineV2
  eval(TASK_ENV, model, observation)
  reset_model(model)
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

# v2 encoder (AttentionPooling)
from features_common.depth_guided_film_online_v2.encoder_film_2model_v2 import DA3Film2ModelEncoderV2
# 共用 backbone 提取器 (无需修改)
from features_common.depth_guided_film_online.extractors_2model import TwoModelExtractors

# 正版 DP
HAS_OFFICIAL_DP = False
try:
    DP_OUTER = Path(features_model_dir) / "third_party" / "DP" / "diffusion_policy"
    if DP_OUTER.exists():
        sys.path.insert(0, str(DP_OUTER))
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        HAS_OFFICIAL_DP = True
        print("[INFO] 正版 DP modules imported")
except ImportError as e:
    print(f"[WARNING] 正版 DP 导入失败: {e}")


# ============================================================
# Policy (与训练脚本中 DA3Film2ModelPolicyV2 完全镜像)
# ============================================================

class Film2ModelDeployPolicyV2(nn.Module):
    """2 模型 DA3-FiLM v2 + proprio + Diffusion Policy (部署版)."""

    def __init__(
        self,
        fusion_encoder: DA3Film2ModelEncoderV2,
        proprio_dim: int = 14,
        action_dim: int = 14,
        horizon: int = 8,
        n_obs_steps: int = 3,
        n_action_steps: int = 6,
        num_inference_steps: int = 100,
    ):
        super().__init__()
        if not HAS_OFFICIAL_DP:
            raise RuntimeError("正版 DP 未加载")

        self.fusion_encoder = fusion_encoder
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.use_proprio = True

        self.normalizer = LinearNormalizer()

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
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    @torch.no_grad()
    def predict_action(self, tokens_list, agent_pos=None):
        B = tokens_list[0].shape[0]
        device = tokens_list[0].device
        fused = self.fusion_encoder(tokens_list)  # [B, To, out_dim]
        if agent_pos is not None:
            nagent_pos = (
                self.normalizer.normalize({"agent_pos": agent_pos})["agent_pos"].to(device)
            )
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

        return self.normalizer.unnormalize({"action": action})["action"]


# ============================================================
# 观测编码
# ============================================================

def encode_obs(observation):
    """将环境观测转换为统一格式."""
    obs = {}
    if "observation" in observation:
        rgb = observation["observation"]["head_camera"]["rgb"]
    else:
        rgb = observation.get("head_camera", {}).get("rgb", None)

    if rgb is None:
        raise ValueError("观测中未找到 head_camera rgb 图像")

    if isinstance(rgb, np.ndarray):
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            rgb = rgb.transpose(2, 0, 1)  # [C, H, W]

    obs["head_cam"] = rgb
    obs["agent_pos"] = observation.get("joint_action", {}).get(
        "vector", np.zeros(14, dtype=np.float32)
    )
    return obs


# ============================================================
# Film2ModelOnlineV2: 完整的模型包装器
# ============================================================

class Film2ModelOnlineV2:
    """2 模型 FiLM v2 在线部署包装器, 兼容 RoBoTwin eval 接口."""

    def __init__(self, usr_args):
        self.usr_args = usr_args
        self.gpu_id = 0
        self.device = torch.device(
            f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.replan_every_call = True
        self.n_action_exec = int(usr_args.get("n_action_exec", 6))

        self.task_name = usr_args["task_name"]
        self.ckpt_setting = usr_args.get("ckpt_setting", "demo_clean")
        self.expert_data_num = usr_args.get("expert_data_num", 50)
        self.seed = usr_args.get("seed", 0)
        self.checkpoint_num = usr_args.get("checkpoint_num", "best")

        # Checkpoint 路径搜索 (v2 目录)
        env_ckpt_dir = os.environ.get("DP2DP3_FILM_ONLINE_V2_CKPT_DIR", "")
        if env_ckpt_dir and Path(env_ckpt_dir).is_dir():
            ckpt_dir = Path(env_ckpt_dir)
            print(f"[Film2ModelV2] Using env ckpt dir: {ckpt_dir}")
        else:
            ckpt_dir_name = (
                f"{self.task_name}-{self.ckpt_setting}-{self.expert_data_num}-{self.seed}"
            )
            ckpt_roots = [
                Path(policy_dir) / "checkpoints_film_online_v2",
                Path(features_model_dir) / "checkpoints_film_online_v2",
            ]
            ckpt_dir = None
            for root in ckpt_roots:
                d = root / ckpt_dir_name
                if d.exists():
                    ckpt_dir = d
                    break
            if ckpt_dir is None:
                raise FileNotFoundError(
                    f"[Film2ModelV2] Checkpoint dir not found. Searched:\n"
                    + "\n".join(f"  {r / ckpt_dir_name}" for r in ckpt_roots)
                )

        # 找到具体的 ckpt 文件
        if str(self.checkpoint_num).lower() == "best":
            best = ckpt_dir / "best.ckpt"
            if best.exists():
                self.ckpt_path = best
            else:
                files = [f for f in ckpt_dir.glob("*.ckpt") if f.stem.isdigit()]
                if not files:
                    raise FileNotFoundError(f"[Film2ModelV2] No .ckpt in {ckpt_dir}")
                self.ckpt_path = max(files, key=lambda f: int(f.stem))
        else:
            self.ckpt_path = ckpt_dir / f"{int(self.checkpoint_num)}.ckpt"
            if not self.ckpt_path.exists():
                raise FileNotFoundError(
                    f"[Film2ModelV2] Checkpoint not found: {self.ckpt_path}"
                )

        print(f"[Film2ModelV2] Loading checkpoint: {self.ckpt_path}")
        self._load_models()

        self.obs_buffer = deque(maxlen=self.n_obs_steps)
        self.action_queue = deque()
        print(
            f"[Film2ModelV2] Ready: horizon={self.horizon}, "
            f"n_obs_steps={self.n_obs_steps}"
        )

    def _load_models(self):
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)

        config = ckpt.get("config", {})
        self.horizon = config.get("data", {}).get("horizon", 8)
        self.n_obs_steps = config.get("data", {}).get("n_obs_steps", 3)
        self.action_dim = 14

        enc_cfg = ckpt.get("encoder_cfg", config.get("encoder", {}))
        # v2: 支持 max_sem_tokens / max_geo_tokens;
        # 兼容旧 ckpt 的 max_tokens 字段
        self.max_sem_tokens = int(
            enc_cfg.get("max_sem_tokens", enc_cfg.get("max_tokens", 196))
        )
        self.max_geo_tokens = int(
            enc_cfg.get("max_geo_tokens", enc_cfg.get("max_tokens", 256))
        )
        print(f"[Film2ModelV2] Encoder config: {enc_cfg}")
        print(f"[Film2ModelV2] max_sem_tokens={self.max_sem_tokens}, "
              f"max_geo_tokens={self.max_geo_tokens}")

        # 1. 加载 DINOv3 + DA3
        print("[Film2ModelV2] Loading DINOv3 + DA3...")
        self.feature_extractors = TwoModelExtractors(gpu_id=self.gpu_id)

        # 2. 创建 v2 Encoder
        print("[Film2ModelV2] Creating DA3Film2ModelEncoderV2...")
        fusion_encoder = DA3Film2ModelEncoderV2(
            semantic_in_dim=int(enc_cfg.get("semantic_in_dim", 768)),
            geometric_in_dim=int(enc_cfg.get("geometric_in_dim", 2048)),
            proj_dim=int(enc_cfg.get("proj_dim", 256)),
            film_hidden=int(enc_cfg.get("film_hidden", 256)),
            out_dim=int(enc_cfg.get("out_dim", 1280)),
            with_pos_enc=bool(enc_cfg.get("with_pos_enc", True)),
            dropout=0.0,  # 推理时关闭 dropout
            max_sem_tokens=self.max_sem_tokens,
            max_geo_tokens=self.max_geo_tokens,
        )

        # 3. 创建 Policy
        print("[Film2ModelV2] Creating Policy...")
        n_action_steps = config.get("data", {}).get("n_action_steps", 6)
        self.policy = Film2ModelDeployPolicyV2(
            fusion_encoder=fusion_encoder,
            proprio_dim=14,
            action_dim=self.action_dim,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=n_action_steps,
            num_inference_steps=config.get("policy", {}).get(
                "num_inference_steps", 100
            ),
        )

        # 4. 加载权重
        print("[Film2ModelV2] Loading weights...")
        state = ckpt.get("policy", {})
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.policy.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARNING] Missing keys: {missing}")
        if unexpected:
            print(f"[WARNING] Unexpected keys: {unexpected}")

        # 5. 加载 normalizer
        if "normalizer" in ckpt:
            self.policy.normalizer.load_state_dict(ckpt["normalizer"])
        else:
            print("[WARNING] Normalizer not found in checkpoint!")

        try:
            self.policy.normalizer.to(self.device)
        except Exception:
            pass

        self.policy = self.policy.to(self.device).eval()
        print("[Film2ModelV2] All models loaded successfully!")

    def reset(self):
        self.obs_buffer.clear()
        self.action_queue.clear()

    def update_obs(self, obs):
        self.obs_buffer.append(obs)

    def pop_action(self):
        if self.action_queue:
            return self.action_queue.popleft()
        return np.zeros(self.action_dim, dtype=np.float32)

    def get_action(self, obs):
        self.obs_buffer.append(obs)
        while len(self.obs_buffer) < self.n_obs_steps:
            self.obs_buffer.append(self.obs_buffer[0])

        if self.replan_every_call or len(self.action_queue) == 0:
            # 1. 准备图像
            images = []
            for o in self.obs_buffer:
                img_np = (o["head_cam"].transpose(1, 2, 0) * 255.0).astype(np.uint8)
                img_np = np.clip(img_np, 0, 255)
                images.append(Image.fromarray(img_np, mode="RGB"))

            # 2. 提取 2 模型 tokens (全量, 再用 linspace 等间隔采样)
            #    与 precompute 保持完全一致的采样方式
            extract_max = None  # 先拿全量 tokens
            tokens_torch = self.feature_extractors.extract_batch_tokens(
                images, max_tokens=extract_max, return_torch=True
            )
            # 用 linspace 等间隔采样, 与训练时 _subsample_tokens 完全一致
            dino_t = self.feature_extractors._subsample_tokens(
                tokens_torch[0], self.max_sem_tokens
            )  # [To, max_sem, 768]
            da3_t = self.feature_extractors._subsample_tokens(
                tokens_torch[1], self.max_geo_tokens
            )  # [To, max_geo, 2048]

            tokens_list = [
                dino_t.float().unsqueeze(0).to(self.device),  # [1, To, max_sem, 768]
                da3_t.float().unsqueeze(0).to(self.device),   # [1, To, max_geo, 2048]
            ]

            # 3. 准备 agent_pos
            agent_pos_tensor = None
            try:
                ap_list = [
                    np.asarray(o.get("agent_pos", []), dtype=np.float32).reshape(-1)
                    for o in self.obs_buffer
                ]
                ap_seq = np.stack(ap_list, axis=0)  # [To, 14]
                agent_pos_tensor = (
                    torch.from_numpy(ap_seq).float().to(self.device).unsqueeze(0)
                )  # [1, To, 14]
            except Exception as e:
                print(f"[WARNING] Failed to prepare agent_pos: {e}")

            # 4. 推理
            with torch.no_grad():
                action_pred = self.policy.predict_action(
                    tokens_list, agent_pos=agent_pos_tensor
                )  # [1, horizon, 14]

            action_pred = action_pred.squeeze(0).cpu().numpy()  # [horizon, 14]
            action_pred = np.clip(action_pred, -3.0, 3.0)

            self.action_queue.clear()
            self.action_queue.extend(action_pred)

        n_exec = min(self.n_action_exec, len(self.action_queue))
        return list(self.action_queue)[:n_exec]


# ============================================================
# RoBoTwin 标准接口
# ============================================================

def get_model(usr_args):
    """RoBoTwin 标准接口: 获取模型."""
    return Film2ModelOnlineV2(usr_args)


def eval(TASK_ENV, model, observation):
    """RoBoTwin 标准接口: 执行推理."""
    obs = encode_obs(observation)
    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        model.pop_action()
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    """RoBoTwin 标准接口: 重置模型."""
    model.reset()
