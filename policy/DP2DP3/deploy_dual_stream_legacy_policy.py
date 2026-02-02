"""DP2DP3 Dual-Stream Legacy Deployment (no token_gate)

用于兼容 train_offline_head1.py 训练的 dual-stream head：
- 不包含 token_gate / token_dropout / ctx_dropout
- _build_cond = global_cond + ctx（与旧训练代码一致）

该模块通过 monkey-patch 复用 deploy_policy.py 的 DP2DP3Model，
避免影响 token-full 推理流程。
"""

import torch
import torch.nn as nn

from . import deploy_policy as base


class LegacyDPRGBDualStreamPolicy(nn.Module):
    """Legacy dual-stream DP policy (no token_gate)."""

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

        if not base.HAS_OFFICIAL_DP:
            raise RuntimeError("正版DP未加载，无法使用LegacyDPRGBDualStreamPolicy")

        self.obs_dim = obs_dim
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        self.normalizer = base.LinearNormalizer()
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

        self.noise_pred_net = base.ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=256,
            diffusion_step_embed_dim=128,
            down_dims=[256, 512, 1024],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
        self.noise_scheduler = base.DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

    def _build_cond(self, obs_global: torch.Tensor, obs_tokens: torch.Tensor) -> torch.Tensor:
        b = obs_global.shape[0]
        obs_flat = obs_global.reshape(b, -1)
        global_cond = self.obs_encoder(obs_flat)
        tokens = obs_tokens.reshape(b, -1, obs_tokens.shape[-1])
        tokens = self.token_proj(tokens)
        query = self.query_proj(global_cond).unsqueeze(1)
        ctx, _ = self.cross_attn(query, tokens, tokens)
        return global_cond + ctx.squeeze(1)

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
                global_cond=obs_cond,
            )
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        if self.use_normalizer:
            action = self.normalizer.unnormalize({'action': action})['action']
        return action


# 替换 deploy_policy 中使用的 DualStream class（仅在本模块作用域内生效）
base.DPRGBDualStreamPolicy = LegacyDPRGBDualStreamPolicy


def get_model(usr_args):
    """使用 deploy_policy.py 的 DP2DP3Model，但内部 DualStream 为 legacy 版本。"""
    return base.DP2DP3Model(usr_args)


# 复用标准接口

eval = base.eval
reset_model = base.reset_model
