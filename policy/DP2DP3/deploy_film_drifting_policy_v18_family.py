"""
DP2DP3 exact-old V18 drifting family deploy.

支持:
  - exact-old V18
  - V18 multi-temp
  - V18 BC-adaptive
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

current_file_path = os.path.abspath(__file__)
policy_dir = os.path.dirname(current_file_path)
features_model_dir = os.path.join(policy_dir, "features_model")
sys.path.insert(0, features_model_dir)

DP_OUTER = Path(features_model_dir) / "third_party" / "DP" / "diffusion_policy"
if DP_OUTER.exists():
    sys.path.insert(0, str(DP_OUTER))

# DP2DP3/__init__.py 会先把主线 features_model 导入进来。
# 为了强制 family deploy 使用 oldv18_repro 里的模块，这里清掉已缓存的 features_common 包。
for module_name in list(sys.modules):
    if module_name == "features_common" or module_name.startswith("features_common."):
        del sys.modules[module_name]

from features_common.depth_guided_film_online.extractors_2model import TwoModelExtractors
from features_common.depth_guided_film_online.encoder_film_2model import DA3Film2ModelEncoder
from features_common.depth_guided_film_drifting.policy_drifting import DA3FilmDriftingPolicy
from features_common.depth_guided_film_drifting.policy_drifting_v18_multitemp import (
    DA3FilmDriftingPolicyV18MultiTemp,
)
from features_common.depth_guided_film_drifting.policy_drifting_v18_bcadapt import (
    DA3FilmDriftingPolicyV18BCAdapt,
)
from features_common.depth_guided_film_drifting.policy_drifting_v18_multitemp_paper import (
    DA3FilmDriftingPolicyV18MultiTempPaper,
)
from features_common.depth_guided_film_drifting.policy_drifting_v18_mtbcadapt import (
    DA3FilmDriftingPolicyV18MTBCAdapt,
)


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


def _build_policy(fusion_encoder, ckpt, config):
    enc_cfg = ckpt.get("encoder_cfg", config.get("encoder", {}))
    drifting_cfg = ckpt.get("drifting_cfg", config.get("drifting", {}))
    policy_class_name = ckpt.get("policy_class", "DA3FilmDriftingPolicy")

    common_kwargs = dict(
        fusion_encoder=fusion_encoder,
        proprio_dim=14,
        action_dim=14,
        horizon=int(config.get("data", {}).get("horizon", 8)),
        n_obs_steps=int(config.get("data", {}).get("n_obs_steps", 3)),
        n_action_steps=int(config.get("data", {}).get("n_action_steps", 6)),
        drifting_temp_scale=float(drifting_cfg.get("temp_scale", 1.0)),
    )

    if policy_class_name == "DA3FilmDriftingPolicyV18MTBCAdapt" or (
        drifting_cfg.get("temp_scales") and float(drifting_cfg.get("bc_lambda", 0.0)) > 0.0
    ):
        policy = DA3FilmDriftingPolicyV18MTBCAdapt(
            drifting_temp_scales=drifting_cfg.get("temp_scales", None),
            drifting_temp_norm_each=bool(drifting_cfg.get("temp_norm_each", True)),
            drifting_temp_norm_eps=float(drifting_cfg.get("temp_norm_eps", 1e-6)),
            bc_lambda=float(drifting_cfg.get("bc_lambda", 0.10)),
            bc_prefix_steps=drifting_cfg.get("bc_prefix_steps", None),
            bc_gate_center=float(drifting_cfg.get("bc_gate_center", 1.0)),
            bc_gate_sharpness=float(drifting_cfg.get("bc_gate_sharpness", 0.25)),
            **common_kwargs,
        )
    elif policy_class_name == "DA3FilmDriftingPolicyV18MultiTempPaper" or drifting_cfg.get("temp_norm_mode") == "paper_global":
        policy = DA3FilmDriftingPolicyV18MultiTempPaper(
            drifting_temp_scales=drifting_cfg.get("temp_scales", None),
            drifting_temp_norm_each=bool(drifting_cfg.get("temp_norm_each", False)),
            drifting_temp_norm_eps=float(drifting_cfg.get("temp_norm_eps", 1e-6)),
            **common_kwargs,
        )
    elif policy_class_name == "DA3FilmDriftingPolicyV18MultiTemp" or drifting_cfg.get("temp_scales"):
        policy = DA3FilmDriftingPolicyV18MultiTemp(
            drifting_temp_scales=drifting_cfg.get("temp_scales", None),
            drifting_temp_norm_each=bool(drifting_cfg.get("temp_norm_each", True)),
            drifting_temp_norm_eps=float(drifting_cfg.get("temp_norm_eps", 1e-6)),
            **common_kwargs,
        )
    elif policy_class_name == "DA3FilmDriftingPolicyV18BCAdapt" or float(drifting_cfg.get("bc_lambda", 0.0)) > 0.0:
        policy = DA3FilmDriftingPolicyV18BCAdapt(
            bc_lambda=float(drifting_cfg.get("bc_lambda", 0.10)),
            bc_prefix_steps=drifting_cfg.get("bc_prefix_steps", None),
            bc_gate_center=float(drifting_cfg.get("bc_gate_center", 1.0)),
            bc_gate_sharpness=float(drifting_cfg.get("bc_gate_sharpness", 0.25)),
            **common_kwargs,
        )
    else:
        policy = DA3FilmDriftingPolicy(**common_kwargs)

    policy.drift_scale = float(drifting_cfg.get("drift_scale", 1.0))
    return policy, enc_cfg, drifting_cfg


@torch.no_grad()
def _predict_action_ensemble(policy, tokens_list, agent_pos, K_ensemble: int):
    K = max(1, int(K_ensemble))
    if K == 1:
        return policy.predict_action(tokens_list, agent_pos=agent_pos)
    preds = []
    for _ in range(K):
        preds.append(policy.predict_action(tokens_list, agent_pos=agent_pos))
    return torch.stack(preds, dim=1).median(dim=1).values


def _factor_grid(k: int) -> Tuple[int, int]:
    if k <= 0:
        return 1, 1
    root = int(np.sqrt(k))
    if root * root == k:
        return root, root
    for h in range(root, 0, -1):
        if k % h == 0:
            return h, k // h
    return 1, k


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _jet_colormap01(x: np.ndarray) -> np.ndarray:
    """x: [H,W] in [0,1] -> [H,W,3] uint8"""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def _token_to_maps(token_kc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """token_kc: [K,C] -> (norm_map[H,W], pca1_map[H,W])"""
    k, c = token_kc.shape
    h, w = _factor_grid(k)
    token_kc = token_kc[: h * w, :]

    norm_map = np.linalg.norm(token_kc, axis=1).reshape(h, w)

    x = token_kc - np.mean(token_kc, axis=0, keepdims=True)
    if c > 1:
        cov = (x.T @ x) / max(1, x.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        p = eigvecs[:, np.argmax(eigvals)]
        pca1 = (x @ p).reshape(h, w)
    else:
        pca1 = x.reshape(h, w)
    return norm_map, pca1


def _save_overlay(base_rgb: np.ndarray, heat01: np.ndarray, out_path: Path, alpha: float = 0.45):
    heat_rgb = _jet_colormap01(heat01)
    base = Image.fromarray(base_rgb)
    heat = Image.fromarray(heat_rgb).resize((base.width, base.height), resample=Image.BILINEAR)
    blend = Image.blend(base, heat, alpha=float(alpha))
    blend.save(out_path)


def _heat_stats(heat01: np.ndarray) -> dict:
    h, w = heat01.shape
    flat_idx = int(np.argmax(heat01))
    py, px = divmod(flat_idx, w)
    mass = float(np.sum(heat01) + 1e-8)
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    cy = float(np.sum(heat01 * yy) / mass)
    cx = float(np.sum(heat01 * xx) / mass)
    return {
        "peak_x": px,
        "peak_y": py,
        "peak_x_norm": float(px / max(1, w - 1)),
        "peak_y_norm": float(py / max(1, h - 1)),
        "com_x": cx,
        "com_y": cy,
        "com_x_norm": float(cx / max(1, w - 1)),
        "com_y_norm": float(cy / max(1, h - 1)),
    }


class FilmDriftingV18FamilyOnline:
    def __init__(self, usr_args):
        self.usr_args = usr_args
        self.gpu_id = 0
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.replan_every_call = True
        self.n_action_exec = int(usr_args.get("n_action_exec", 6))
        self.task_name = usr_args["task_name"]
        self.ckpt_setting = usr_args.get("ckpt_setting", "demo_clean-v18_exactold")
        self.expert_data_num = usr_args.get("expert_data_num", 50)
        self.seed = usr_args.get("seed", 0)
        self.checkpoint_num = usr_args.get("checkpoint_num", "best")

        env_ckpt_dir = (
            os.environ.get("DP2DP3_FILM_DRIFTING_V18_CKPT_DIR", "")
            or os.environ.get("DP2DP3_FILM_DRIFTING_CKPT_DIR", "")
        )
        if env_ckpt_dir and Path(env_ckpt_dir).is_dir():
            ckpt_dir = Path(env_ckpt_dir)
            print(f"[FilmDriftingV18] Using env ckpt dir: {ckpt_dir}")
        else:
            ckpt_dir_name = (
                f"{self.task_name}-{self.ckpt_setting}-{self.expert_data_num}-{self.seed}"
            )
            ckpt_roots = [Path(policy_dir) / "checkpoints_film_drifting"]
            ckpt_dir = None
            for root in ckpt_roots:
                d = root / ckpt_dir_name
                if d.exists():
                    ckpt_dir = d
                    break
            if ckpt_dir is None:
                raise FileNotFoundError(
                    "[FilmDriftingV18] Checkpoint dir not found. Searched:\n"
                    + "\n".join(f"  {r / ckpt_dir_name}" for r in ckpt_roots)
                )

        ckpt_selector = str(self.checkpoint_num).lower()
        if ckpt_selector == "best":
            best = ckpt_dir / "best.ckpt"
            if best.exists():
                self.ckpt_path = best
            else:
                files = [f for f in ckpt_dir.glob("*.ckpt") if f.stem.isdigit()]
                if not files:
                    raise FileNotFoundError(f"[FilmDriftingV18] No .ckpt in {ckpt_dir}")
                self.ckpt_path = max(files, key=lambda f: int(f.stem))
        else:
            self.ckpt_path = ckpt_dir / f"{int(self.checkpoint_num)}.ckpt"
            if not self.ckpt_path.exists():
                raise FileNotFoundError(f"[FilmDriftingV18] Checkpoint not found: {self.ckpt_path}")

        print(f"[FilmDriftingV18] Loading checkpoint: {self.ckpt_path}")
        self._load_models()

        self.K_ensemble = int(usr_args.get("K_ensemble", 1))
        self.obs_buffer = deque(maxlen=self.n_obs_steps)
        self.action_queue = deque()
        self._timing_backbone_ms = []
        self._timing_policy_ms = []
        self._timing_total_ms = []
        self._episode_call_counts = []
        self._episode_results = []
        self._current_episode_calls = 0

        self.vis_dump_dir = usr_args.get("vis_dump_dir", None)
        self.vis_dump_max_calls = int(usr_args.get("vis_dump_max_calls", 0))
        self.vis_dump_alpha = float(usr_args.get("vis_dump_alpha", 0.45))
        self._vis_dump_calls = 0
        if self.vis_dump_dir:
            Path(self.vis_dump_dir).mkdir(parents=True, exist_ok=True)
            print(f"[FilmDriftingV18] Visual dump enabled: {self.vis_dump_dir}")

        print(
            f"[FilmDriftingV18] Ready: horizon={self.horizon}, "
            f"n_obs_steps={self.n_obs_steps}, K_ensemble={self.K_ensemble}"
        )

    def _load_models(self):
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        self.horizon = config.get("data", {}).get("horizon", 8)
        self.n_obs_steps = config.get("data", {}).get("n_obs_steps", 3)
        self.action_dim = 14
        n_action_steps = config.get("data", {}).get("n_action_steps", 6)

        enc_cfg = ckpt.get("encoder_cfg", config.get("encoder", {}))
        self.max_tokens = int(enc_cfg.get("max_tokens", 196))

        self.feature_extractors = TwoModelExtractors(gpu_id=self.gpu_id)
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

        self.policy, _, drifting_cfg = _build_policy(fusion_encoder, ckpt, config)
        print(f"[FilmDriftingV18] Encoder config: {enc_cfg}")
        print(f"[FilmDriftingV18] Drifting config: {drifting_cfg}")

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
        try:
            self.policy.normalizer.to(self.device)
        except Exception:
            pass

        self.policy = self.policy.to(self.device).eval()
        print("[FilmDriftingV18] All models loaded!")

    def reset(self):
        if self._current_episode_calls > 0:
            self._episode_call_counts.append(self._current_episode_calls)
        self._current_episode_calls = 0
        self.obs_buffer.clear()
        self.action_queue.clear()

    def mark_episode_result(self, success):
        self._episode_results.append(success)

    def get_timing_summary(self):
        if self._current_episode_calls > 0:
            self._episode_call_counts.append(self._current_episode_calls)
            self._current_episode_calls = 0
        if not self._timing_total_ms:
            return ""
        n = len(self._timing_total_ms)
        skip = min(1, n - 1)
        bb = self._timing_backbone_ms[skip:]
        pol = self._timing_policy_ms[skip:]
        tot = self._timing_total_ms[skip:]
        if not tot:
            return ""
        lines = [
            f"\n--- Inference Timing ({len(tot)} calls, excl. {skip} warmup) ---",
            f"Per-call backbone (DINOv3+DA3):  mean={np.mean(bb):.1f}ms  std={np.std(bb):.1f}ms",
            f"Per-call policy (Encoder+UNet):   mean={np.mean(pol):.1f}ms  std={np.std(pol):.1f}ms",
            f"Per-call total:                   mean={np.mean(tot):.1f}ms  std={np.std(tot):.1f}ms",
            "Action Head: V18 Drifting Family (1 NFE)",
        ]
        return "\n".join(lines)

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

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_bb_start = time.perf_counter()

            tokens_torch = self.feature_extractors.extract_batch_tokens(
                images, max_tokens=self.max_tokens, return_torch=True
            )
            tokens_list = [t.float().unsqueeze(0).to(self.device) for t in tokens_torch]

            if self.vis_dump_dir and (
                self.vis_dump_max_calls <= 0 or self._vis_dump_calls < self.vis_dump_max_calls
            ):
                try:
                    self._dump_visual_maps(images, tokens_torch)
                    self._vis_dump_calls += 1
                except Exception as e:
                    print(f"[FilmDriftingV18][WARN] visual dump failed: {e}")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_bb_end = time.perf_counter()

            agent_pos_tensor = None
            try:
                ap_list = [
                    np.asarray(o.get("agent_pos", []), dtype=np.float32).reshape(-1)
                    for o in self.obs_buffer
                ]
                ap_seq = np.stack(ap_list, axis=0)
                agent_pos_tensor = torch.from_numpy(ap_seq).float().to(self.device).unsqueeze(0)
            except Exception as e:
                print(f"[WARNING] agent_pos error: {e}")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_pol_start = time.perf_counter()

            with torch.no_grad():
                action_pred = _predict_action_ensemble(
                    self.policy,
                    tokens_list,
                    agent_pos_tensor,
                    self.K_ensemble,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_pol_end = time.perf_counter()

            bb_ms = (t_bb_end - t_bb_start) * 1000
            pol_ms = (t_pol_end - t_pol_start) * 1000
            total_ms = bb_ms + pol_ms
            self._timing_backbone_ms.append(bb_ms)
            self._timing_policy_ms.append(pol_ms)
            self._timing_total_ms.append(total_ms)
            self._current_episode_calls += 1

            action_pred = action_pred.squeeze(0).cpu().numpy()
            action_pred = np.clip(action_pred, -3.0, 3.0)
            self.action_queue.clear()
            self.action_queue.extend(action_pred)

        n_exec = min(self.n_action_exec, len(self.action_queue))
        return list(self.action_queue)[:n_exec]

    def _dump_visual_maps(self, images, tokens_torch):
        out_dir = Path(self.vis_dump_dir)
        model_names = ["dino", "da3"]
        call_id = self._vis_dump_calls
        stats_path = out_dir / "feature_stats.csv"
        if not stats_path.exists():
            with open(stats_path, "w", encoding="utf-8") as f:
                f.write(
                    "call_id,obs_id,model,map_type,peak_x,peak_y,peak_x_norm,peak_y_norm,com_x,com_y,com_x_norm,com_y_norm\n"
                )

        for m_idx, t_bkc in enumerate(tokens_torch):
            if not isinstance(t_bkc, torch.Tensor):
                continue
            if t_bkc.ndim != 3:
                continue
            b, _, _ = t_bkc.shape
            mname = model_names[m_idx] if m_idx < len(model_names) else f"model{m_idx}"

            for bi in range(b):
                base_rgb = np.asarray(images[bi].convert("RGB"))
                token_kc = t_bkc[bi].detach().cpu().numpy()
                norm_map, pca1_map = _token_to_maps(token_kc)
                norm01 = _normalize_map(norm_map)
                pca01 = _normalize_map(pca1_map)

                norm_path = out_dir / f"call{call_id:04d}_obs{bi}_{mname}_norm_overlay.jpg"
                pca_path = out_dir / f"call{call_id:04d}_obs{bi}_{mname}_pca1_overlay.jpg"
                _save_overlay(base_rgb, norm01, norm_path, alpha=self.vis_dump_alpha)
                _save_overlay(base_rgb, pca01, pca_path, alpha=self.vis_dump_alpha)

                norm_s = _heat_stats(norm01)
                pca_s = _heat_stats(pca01)
                with open(stats_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{call_id},{bi},{mname},norm,{norm_s['peak_x']},{norm_s['peak_y']},{norm_s['peak_x_norm']:.6f},{norm_s['peak_y_norm']:.6f},{norm_s['com_x']:.6f},{norm_s['com_y']:.6f},{norm_s['com_x_norm']:.6f},{norm_s['com_y_norm']:.6f}\n"
                    )
                    f.write(
                        f"{call_id},{bi},{mname},pca1,{pca_s['peak_x']},{pca_s['peak_y']},{pca_s['peak_x_norm']:.6f},{pca_s['peak_y_norm']:.6f},{pca_s['com_x']:.6f},{pca_s['com_y']:.6f},{pca_s['com_x_norm']:.6f},{pca_s['com_y_norm']:.6f}\n"
                    )


def get_model(usr_args):
    return FilmDriftingV18FamilyOnline(usr_args)


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
