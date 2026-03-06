"""DP2DP3 deploy_dino_online_policy.py

消融实验: 纯 DINOv3 部署脚本 (无 DA3)
======================================

架构说明:
  - 只加载 DINOv3 backbone (最省显存)
  - Encoder: DinoOnlyEncoder (无调制)
  - Policy:  Diffusion Policy (与 FiLM 版完全一致的 DP head)
  - 推理: RGB -> DINOv3 tokens -> DinoOnlyEncoder -> DP -> action

RoBoTwin 标准接口:
  get_model(usr_args) -> DinoOnlineModel
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

current_file_path = os.path.abspath(__file__)
policy_dir = os.path.dirname(current_file_path)
features_model_dir = os.path.join(policy_dir, "features_model")
sys.path.insert(0, features_model_dir)

from PIL import Image

from features_common.depth_guided_dino_online.extractors_dino_only import DinoOnlyExtractors
from features_common.depth_guided_dino_online.encoder_dino_only import DinoOnlyEncoder

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
# Policy (与训练脚本 DinoOnlyPolicy 镜像)
# ============================================================

class DinoOnlineDeployPolicy(nn.Module):
    def __init__(
        self,
        fusion_encoder: DinoOnlyEncoder,
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
        fused = self.fusion_encoder(tokens_list)   # [B, To, out_dim]
        if agent_pos is not None:
            nagent_pos = (
                self.normalizer.normalize({"agent_pos": agent_pos})["agent_pos"].to(device)
            )
            obs_combined = torch.cat([fused, nagent_pos], dim=-1)
        else:
            zeros = torch.zeros(B, fused.shape[1], self.proprio_dim, device=device)
            obs_combined = torch.cat([fused, zeros], dim=-1)

        obs_cond = self.obs_encoder(obs_combined.reshape(B, -1))
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
            rgb = rgb.transpose(2, 0, 1)
    obs["head_cam"] = rgb
    obs["agent_pos"] = observation.get("joint_action", {}).get(
        "vector", np.zeros(14, dtype=np.float32)
    )
    return obs


# ============================================================
# DinoOnlineModel
# ============================================================

class DinoOnlineModel:
    """纯 DINOv3 在线部署包装器, 兼容 RoBoTwin eval 接口."""

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

        env_ckpt_dir = os.environ.get("DP2DP3_DINO_ONLINE_CKPT_DIR", "")
        if env_ckpt_dir and Path(env_ckpt_dir).is_dir():
            ckpt_dir = Path(env_ckpt_dir)
        else:
            ckpt_dir_name = (
                f"{self.task_name}-{self.ckpt_setting}-{self.expert_data_num}-{self.seed}"
            )
            ckpt_roots = [
                Path(policy_dir) / "checkpoints_dino_online",
                Path(features_model_dir) / "checkpoints_dino_online",
            ]
            ckpt_dir = None
            for root in ckpt_roots:
                d = root / ckpt_dir_name
                if d.exists():
                    ckpt_dir = d
                    break
            if ckpt_dir is None:
                raise FileNotFoundError(
                    f"[DinoOnline] Checkpoint dir not found. Searched:\n"
                    + "\n".join(f"  {r / ckpt_dir_name}" for r in ckpt_roots)
                )

        if str(self.checkpoint_num).lower() == "best":
            best = ckpt_dir / "best.ckpt"
            if best.exists():
                self.ckpt_path = best
            else:
                files = [f for f in ckpt_dir.glob("*.ckpt") if f.stem.isdigit()]
                if not files:
                    raise FileNotFoundError(f"[DinoOnline] No .ckpt in {ckpt_dir}")
                self.ckpt_path = max(files, key=lambda f: int(f.stem))
        else:
            self.ckpt_path = ckpt_dir / f"{int(self.checkpoint_num)}.ckpt"
            if not self.ckpt_path.exists():
                raise FileNotFoundError(f"[DinoOnline] Checkpoint not found: {self.ckpt_path}")

        print(f"[DinoOnline] Loading checkpoint: {self.ckpt_path}")
        self._load_models()

        self.obs_buffer = deque(maxlen=self.n_obs_steps)
        self.action_queue = deque()
        print(f"[DinoOnline] Ready: horizon={self.horizon}, n_obs_steps={self.n_obs_steps}")

    def _load_models(self):
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)

        config = ckpt.get("config", {})
        self.horizon = config.get("data", {}).get("horizon", 8)
        self.n_obs_steps = config.get("data", {}).get("n_obs_steps", 3)
        self.action_dim = 14

        enc_cfg = ckpt.get("encoder_cfg", config.get("encoder", {}))
        self.max_tokens = int(enc_cfg.get("max_tokens", 196))
        print(f"[DinoOnline] Encoder config: {enc_cfg}")

        print("[DinoOnline] Loading DINOv3...")
        self.feature_extractors = DinoOnlyExtractors(gpu_id=self.gpu_id)

        print("[DinoOnline] Creating DinoOnlyEncoder...")
        fusion_encoder = DinoOnlyEncoder(
            semantic_in_dim=int(enc_cfg.get("semantic_in_dim", 768)),
            proj_dim=int(enc_cfg.get("proj_dim", 256)),
            out_dim=int(enc_cfg.get("out_dim", 1280)),
            with_pos_enc=bool(enc_cfg.get("with_pos_enc", True)),
            dropout=0.0,
            max_tokens=self.max_tokens,
        )

        n_action_steps = config.get("data", {}).get("n_action_steps", 6)
        self.policy = DinoOnlineDeployPolicy(
            fusion_encoder=fusion_encoder,
            proprio_dim=14,
            action_dim=self.action_dim,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=n_action_steps,
            num_inference_steps=config.get("policy", {}).get("num_inference_steps", 100),
        )

        state = ckpt.get("policy", {})
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.policy.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARNING] Missing keys: {missing}")
        if unexpected:
            print(f"[WARNING] Unexpected keys: {unexpected}")

        if "normalizer" in ckpt:
            self.policy.normalizer.load_state_dict(ckpt["normalizer"])
        else:
            print("[WARNING] Normalizer not in checkpoint!")

        try:
            self.policy.normalizer.to(self.device)
        except Exception:
            pass

        self.policy = self.policy.to(self.device).eval()
        print("[DinoOnline] All models loaded!")

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
            images = []
            for o in self.obs_buffer:
                img_np = (o["head_cam"].transpose(1, 2, 0) * 255.0).astype(np.uint8)
                img_np = np.clip(img_np, 0, 255)
                images.append(Image.fromarray(img_np, mode="RGB"))

            tokens_torch = self.feature_extractors.extract_batch_tokens(
                images, max_tokens=self.max_tokens, return_torch=True
            )
            tokens_list = [
                t.float().unsqueeze(0).to(self.device)
                for t in tokens_torch
            ]

            agent_pos_tensor = None
            try:
                ap_list = [
                    np.asarray(o.get("agent_pos", []), dtype=np.float32).reshape(-1)
                    for o in self.obs_buffer
                ]
                ap_seq = np.stack(ap_list, axis=0)
                agent_pos_tensor = (
                    torch.from_numpy(ap_seq).float().to(self.device).unsqueeze(0)
                )
            except Exception as e:
                print(f"[WARNING] agent_pos 处理失败: {e}")

            with torch.no_grad():
                action_pred = self.policy.predict_action(
                    tokens_list, agent_pos=agent_pos_tensor
                )

            action_pred = action_pred.squeeze(0).cpu().numpy()
            action_pred = np.clip(action_pred, -3.0, 3.0)

            self.action_queue.clear()
            self.action_queue.extend(action_pred)

        n_exec = min(self.n_action_exec, len(self.action_queue))
        return list(self.action_queue)[:n_exec]


# ============================================================
# RoBoTwin 标准接口
# ============================================================

def get_model(usr_args):
    return DinoOnlineModel(usr_args)


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    actions = model.get_action(obs)
    for action in actions:
        TASK_ENV.take_action(action)
        model.pop_action()
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.reset()
