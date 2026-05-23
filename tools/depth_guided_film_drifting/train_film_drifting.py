#!/usr/bin/env python3
"""RoboTwin 根目录兼容入口: 转发到 DP2DP3 features_model 训练脚本。"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_config(config_arg: str, repo_root: Path, fm_root: Path) -> str:
    config_path = Path(config_arg)
    if config_path.exists():
        return str(config_path)

    mapped = fm_root / config_path
    if mapped.exists():
        return str(mapped)

    if config_arg.startswith("configs/"):
        mapped = fm_root / config_arg
        if mapped.exists():
            return str(mapped)

    return config_arg


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fm_root = repo_root / "policy" / "DP2DP3" / "features_model"
    target = fm_root / "tools" / "depth_guided_film_drifting" / "train_film_drifting.py"

    if not target.exists():
        raise FileNotFoundError(f"目标训练脚本不存在: {target}")

    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            argv[i + 1] = _resolve_config(argv[i + 1], repo_root, fm_root)
            break

    os.execv(sys.executable, [sys.executable, str(target), *argv])


if __name__ == "__main__":
    main()
