import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from termcolor import cprint


# 将 ULIP-main 加入搜索路径，方便直接复用官方 PointBERT 编码器
ULIP_ROOT = Path(__file__).resolve().parents[4] / "ULIP-main"
if str(ULIP_ROOT) not in sys.path:
    sys.path.append(str(ULIP_ROOT))

try:
    from models.pointbert.point_encoder import PointTransformer, PointTransformer_Colored
except Exception as exc:  # pragma: no cover - 仅在依赖缺失时触发
    raise ImportError(
        "无法导入 ULIP PointBERT 编码器，请确认 `policy/DP3/ULIP-main` 在路径上并已安装依赖。"
    ) from exc


@dataclass
class ULIPBackboneConfig:
    """与 ULIP-2 PointBERT 10k 配置保持一致的默认参数。"""

    trans_dim: int = 384
    depth: int = 18
    drop_path_rate: float = 0.1
    cls_dim: int = 40
    num_heads: int = 6
    group_size: int = 32
    num_group: int = 512
    encoder_dims: int = 256


class _DummyArgs:
    """PointBERT 需要的简单 args 容器，仅使用 evaluate_3d 字段。"""

    def __init__(self, evaluate_3d: bool = True) -> None:
        self.evaluate_3d = evaluate_3d


def _freeze_module(module: nn.Module) -> None:
    """冻结一个 module 的全部参数。"""
    for param in module.parameters():
        param.requires_grad = False


def _build_projector(
    in_dim: int, out_dim: int, final_norm: str = "layernorm"
) -> nn.Module:
    """统一将 ULIP 特征映射到 DP3 动作头所需维度。"""
    layers = [nn.Linear(in_dim, out_dim)]
    if final_norm == "layernorm":
        layers.append(nn.LayerNorm(out_dim))
    elif final_norm == "none":
        pass
    else:
        raise NotImplementedError(f"final_norm: {final_norm}")
    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


def _clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """提取 checkpoint 中与 point_encoder 相关的权重。"""
    cleaned = {}
    for key, value in state_dict.items():
        if "point_encoder." in key:
            new_key = key.split("point_encoder.", 1)[1]
        elif key.startswith("module."):
            new_key = key[len("module.") :]
        else:
            new_key = key
        cleaned[new_key] = value
    return cleaned


class _ULIPEncoderBase(nn.Module):
    """封装 ULIP PointBERT 编码器，输出与 DP3 动作头对齐的特征。"""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        training_mode: str = "frozen",
        pretrained_model_path: Optional[str] = None,
        ulip_cfg: Optional[Dict[str, Any]] = None,
        final_norm: str = "layernorm",
        strict_load: bool = False,
        colored: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()

        training_mode = training_mode.lower()
        # 验证训练模式与输入通道数
        assert training_mode in {"frozen", "finetune", "scratch"}, (
            f"Unsupported training_mode {training_mode}"
        )
        assert in_channels == (6 if colored else 3), (
            f"ULIPEncoder expects {6 if colored else 3} input channels, got {in_channels}"
        )
        # 构建 ULIP PointBERT 编码器
        cfg_dict = asdict(ULIPBackboneConfig())
        if ulip_cfg is not None:
            cfg_dict.update(ulip_cfg)
        cfg = ULIPBackboneConfig(**cfg_dict)
        # 创建 PointBERT backbone
        backbone_cls = PointTransformer_Colored if colored else PointTransformer
        self.backbone = backbone_cls(cfg, args=_DummyArgs(evaluate_3d=True))
        # 创建投影层
        self.feature_dim = cfg.trans_dim * 2  # PointBERT 输出 concat(cls, max_pool)# PointBERT 输出 768 维
        self.projector = _build_projector(
            in_dim=self.feature_dim, 
            out_dim=out_channels,   # 投影到 DP3 需要的维度（如 128）
            final_norm=final_norm
        )

        cprint(
            f"[ULIPEncoder] mode={training_mode}, "
            f"ckpt={'None' if pretrained_model_path is None else pretrained_model_path}, "
            f"in={in_channels}, out={out_channels}, feat_dim={self.feature_dim}",
            "cyan",
        )

        self._backbone_frozen = False
        self._maintain_eval_stats = False

        if training_mode in {"frozen", "finetune"}:
            if pretrained_model_path:
                self._load_pretrained(pretrained_model_path, strict=strict_load)
            _freeze_module(self.backbone)
            self._set_backbone_frozen_state()
            if training_mode == "frozen":
                cprint("[ULIPEncoder] Backbone frozen (frozen mode)", "yellow")
            else:
                cprint("[ULIPEncoder] Backbone initially frozen (finetune mode, will unfreeze later)", "yellow")
        elif training_mode == "scratch":
            cprint("[ULIPEncoder] Training from scratch (不加载预训练权重)", "magenta")
        else:
            cprint("[ULIPEncoder] 未提供预训练权重,继续初始化权重", "yellow")

    def _load_pretrained(self, path: str, strict: bool = False) -> None:
        # 从 checkpoint 加载预训练权重
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        if not isinstance(state_dict, dict):
            cprint(f"[ULIPEncoder] 无法解析 checkpoint: {path}", "red")
            return

        backbone_state = _clean_state_dict(state_dict)
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=strict)

        cprint(f"[ULIPEncoder] 加载预训练权重自 {path}", "green")
        cprint(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}", "yellow")
        cprint(f"  missing keys: {missing}", "yellow")  # 看看是哪些参数缺失
        cprint(f"  unexpected keys: {unexpected}", "yellow")  # 看看是哪些多余参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) -> feat: (B, feat_dim) -> (B, out_channels)
        # 确保输入是连续的，避免 pointnet2_ops 的 contiguous 断言错误
        x = x.contiguous()
        feat = self.backbone(x)
        return self.projector(feat)

    def _set_backbone_frozen_state(self, frozen: bool = True):
        self._backbone_frozen = frozen
        self._maintain_eval_stats = frozen
        if frozen:
            self.backbone.eval()
        else:
            self.backbone.train()

    def set_backbone_train_mode(self, train_backbone: bool):
        """显式切换 backbone 的统计模式，用于 finetune 解冻。"""
        self._set_backbone_frozen_state(not train_backbone)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._maintain_eval_stats:
            self.backbone.eval()
        return self


class ULIPEncoderXYZ(_ULIPEncoderBase):
    """仅 xyz 的 ULIP-PointBERT 编码器。"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        training_mode: str = "frozen",
        pretrained_model_path: Optional[str] = None,
        ulip_cfg: Optional[Dict[str, Any]] = None,
        final_norm: str = "layernorm",
        strict_load: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            training_mode=training_mode,
            pretrained_model_path=pretrained_model_path,
            ulip_cfg=ulip_cfg,
            final_norm=final_norm,
            strict_load=strict_load,
            colored=False,
            **kwargs,
        )


class ULIPEncoderXYZRGB(_ULIPEncoderBase):
    """xyz + rgb 的 ULIP-PointBERT 编码器。"""

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 256,
        training_mode: str = "frozen",
        pretrained_model_path: Optional[str] = None,
        ulip_cfg: Optional[Dict[str, Any]] = None,
        final_norm: str = "layernorm",
        strict_load: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            training_mode=training_mode,
            pretrained_model_path=pretrained_model_path,
            ulip_cfg=ulip_cfg,
            final_norm=final_norm,
            strict_load=strict_load,
            colored=True,
            **kwargs,
        )
