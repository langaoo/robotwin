"""DP2DP3 deploy_film_drifting_policy.py

DA3-FiLM + Drifting Head 部署脚本 (Innovation 2)
=================================================

架构:
  - DINOv3 + DA3 backbone (完全相同于 film_online)
  - Encoder: DA3Film2ModelEncoder (FiLM 调制)
  - Head: DriftingActionGenerator (单步生成，1 NFE)
  - 推理: RGB -> tokens -> FiLM -> Drifting generator -> action

与 deploy_film_online_policy.py 的唯一区别:
  - Policy 类为 DA3FilmDriftingPolicy（无 DDPM，无迭代去噪）
  - 推理时只需 1 次前向传播
  - 环境变量: DP2DP3_FILM_DRIFTING_CKPT_DIR

RoBoTwin 标准接口:
  get_model(usr_args) -> FilmDriftingOnline
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

current_file_path  = os.path.abspath(__file__)
policy_dir         = os.path.dirname(current_file_path)
features_model_dir = os.path.join(policy_dir, "features_model")
sys.path.insert(0, features_model_dir)

from PIL import Image

# DP 路径（normalizer）：features_model/DP/diffusion_policy/ 是 git repo 根目录
# 其内部的 diffusion_policy/ 才是 Python package，需要添加 repo 根目录到 sys.path
DP_OUTER = Path(features_model_dir) / "DP" / "diffusion_policy"
if DP_OUTER.exists():
    sys.path.insert(0, str(DP_OUTER))

from features_common.depth_guided_film_online.extractors_2model import TwoModelExtractors
from features_common.depth_guided_film_online.encoder_film_2model import DA3Film2ModelEncoder
from features_common.depth_guided_film_drifting.policy_drifting import DA3FilmDriftingPolicy


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


class FilmDriftingOnline:
    """DA3-FiLM + Drifting Head 在线部署包装器."""

    def __init__(self, usr_args):
        self.usr_args    = usr_args
        self.gpu_id      = 0
        self.device      = torch.device(
            f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.replan_every_call = True
        self.n_action_exec     = int(usr_args.get("n_action_exec", 6))
        self.task_name         = usr_args["task_name"]
        self.ckpt_setting      = usr_args.get("ckpt_setting", "demo_clean")
        self.expert_data_num   = usr_args.get("expert_data_num", 50)
        self.seed              = usr_args.get("seed", 0)
        self.checkpoint_num    = usr_args.get("checkpoint_num", "best")

        # ---- Checkpoint dir 搜索 ----
        env_ckpt_dir = os.environ.get("DP2DP3_FILM_DRIFTING_CKPT_DIR", "")
        if env_ckpt_dir and Path(env_ckpt_dir).is_dir():
            ckpt_dir = Path(env_ckpt_dir)
            print(f"[FilmDrifting] Using env ckpt dir: {ckpt_dir}")
        else:
            ckpt_dir_name = (
                f"{self.task_name}-{self.ckpt_setting}-"
                f"{self.expert_data_num}-{self.seed}"
            )
            ckpt_roots = [
                Path(policy_dir) / "checkpoints_film_drifting",
                Path(features_model_dir) / "checkpoints_film_drifting",
            ]
            ckpt_dir = None
            for root in ckpt_roots:
                d = root / ckpt_dir_name
                if d.exists():
                    ckpt_dir = d
                    break
            if ckpt_dir is None:
                raise FileNotFoundError(
                    f"[FilmDrifting] Checkpoint dir not found. Searched:\n"
                    + "\n".join(f"  {r / ckpt_dir_name}" for r in ckpt_roots)
                )

        # ---- 具体 ckpt 文件 ----
        if str(self.checkpoint_num).lower() == "best":
            best = ckpt_dir / "best.ckpt"
            if best.exists():
                self.ckpt_path = best
            else:
                files = [f for f in ckpt_dir.glob("*.ckpt") if f.stem.isdigit()]
                if not files:
                    raise FileNotFoundError(f"[FilmDrifting] No .ckpt in {ckpt_dir}")
                self.ckpt_path = max(files, key=lambda f: int(f.stem))
        else:
            self.ckpt_path = ckpt_dir / f"{int(self.checkpoint_num)}.ckpt"
            if not self.ckpt_path.exists():
                raise FileNotFoundError(
                    f"[FilmDrifting] Checkpoint not found: {self.ckpt_path}"
                )

        print(f"[FilmDrifting] Loading checkpoint: {self.ckpt_path}")
        self._load_models()

        self.obs_buffer   = deque(maxlen=self.n_obs_steps)
        self.action_queue = deque()
        print(
            f"[FilmDrifting] Ready: horizon={self.horizon}, "
            f"n_obs_steps={self.n_obs_steps}  (1-step inference)"
        )

    def _load_models(self):
        ckpt   = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})

        self.horizon      = config.get("data", {}).get("horizon", 8)
        self.n_obs_steps  = config.get("data", {}).get("n_obs_steps", 3)
        self.action_dim   = 14
        n_action_steps    = config.get("data", {}).get("n_action_steps", 6)

        enc_cfg         = ckpt.get("encoder_cfg", config.get("encoder", {}))
        drifting_cfg    = ckpt.get("drifting_cfg", config.get("drifting", {}))
        self.max_tokens = int(enc_cfg.get("max_tokens", 196))

        print(f"[FilmDrifting] Encoder config: {enc_cfg}")
        print(f"[FilmDrifting] Drifting config: {drifting_cfg}")

        # 1. Backbones
        self.feature_extractors = TwoModelExtractors(gpu_id=self.gpu_id)

        # 2. Encoder
        fusion_encoder = DA3Film2ModelEncoder(
            semantic_in_dim=int(enc_cfg.get("semantic_in_dim", 768)),
            geometric_in_dim=int(enc_cfg.get("geometric_in_dim", 2048)),
            proj_dim=int(enc_cfg.get("proj_dim", 256)),
            film_hidden=int(enc_cfg.get("film_hidden", 256)),
            out_dim=int(enc_cfg.get("out_dim", 1280)),
            with_pos_enc=bool(enc_cfg.get("with_pos_enc", True)),
            dropout=0.0,
            max_tokens=self.max_tokens,
        )

        # 3. ★ Drifting Policy (no DDPM)
        self.policy = DA3FilmDriftingPolicy(
            fusion_encoder=fusion_encoder,
            proprio_dim=14,
            action_dim=self.action_dim,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=n_action_steps,
            drifting_temp=float(drifting_cfg.get("temp", 0.05)),
            hidden_dim=int(drifting_cfg.get("hidden_dim", 1024)),
        )

        # 4. 加载权重
        state = ckpt.get("policy", {})
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.policy.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARNING] Missing keys: {missing}")
        if unexpected:
            print(f"[WARNING] Unexpected keys: {unexpected}")

        # 5. Normalizer
        if "normalizer" in ckpt:
            self.policy.normalizer.load_state_dict(ckpt["normalizer"])
        else:
            print("[WARNING] Normalizer not found!")
        try:
            self.policy.normalizer.to(self.device)
        except Exception:
            pass

        self.policy = self.policy.to(self.device).eval()
        print("[FilmDrifting] All models loaded!")

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
                print(f"[WARNING] agent_pos error: {e}")

            with torch.no_grad():
                action_pred = self.policy.predict_action(
                    tokens_list, agent_pos=agent_pos_tensor
                )  # [1, horizon, 14]

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
    return FilmDriftingOnline(usr_args)


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
