#!/usr/bin/env python3
"""
对比两次 DP2DP3 推理日志（JSONL），量化并可视化“闪现/卡顿/回跳”：

输入日志来自 policy/DP2DP3/deploy_policy.py 的 action_log_path（stage=exec）。
要求 exec 记录包含：
  - tcp_after: {"left": [...], "right": [...]}（至少 xyz）
  - episode: int

用法示例：
  /home/gl/miniconda3/envs/RoboTwin/bin/python policy/DP2DP3/plot_trajectory_compare.py \
    --log_a policy/DP2DP3/logs/action_logs/pool_ws1.jsonl --name_a pool_ws1 \
    --log_b policy/DP2DP3/logs/action_logs/dual_stream.jsonl --name_b dual_stream \
    --arm right --episode 0 --out_dir policy/DP2DP3/logs/trajectory_plots
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _percentiles(x: np.ndarray, ps: Tuple[float, ...] = (50, 90, 95, 99, 99.5, 99.9)) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{p:g}": float("nan") for p in ps}
    vals = np.percentile(x, list(ps))
    return {f"p{p:g}": float(v) for p, v in zip(ps, vals)}


def _extract_tcp_xyz(r: Dict[str, Any], arm: str) -> Optional[np.ndarray]:
    tcp = r.get("tcp_after")
    if not isinstance(tcp, dict):
        return None
    vec = tcp.get(arm)
    if vec is None:
        return None
    try:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            return None
        return arr[:3]
    except Exception:
        return None


@dataclass
class TeleportSummary:
    log: str
    name: str
    arm: str
    episode: int
    n_steps: int
    step_disp_m: Dict[str, float]
    step_disp_top: List[Dict[str, Any]]
    rewind_teleports: int
    rewind_examples: List[Dict[str, Any]]


def analyze_log(
    rows: List[Dict[str, Any]],
    log_path: Path,
    name: str,
    arm: str,
    episode: int,
    topk: int,
    teleport_thresh: float,
    rewind_return_thresh: float,
    rewind_window_min: int,
    rewind_window_max: int,
) -> Tuple[TeleportSummary, np.ndarray]:
    execs = [r for r in rows if r.get("stage") == "exec" and int(r.get("episode", -1)) == int(episode)]
    execs = sorted(execs, key=lambda x: int(x.get("step", 0)))
    xyzs: List[np.ndarray] = []
    kept: List[Dict[str, Any]] = []
    for r in execs:
        xyz = _extract_tcp_xyz(r, arm=arm)
        if xyz is None or not np.all(np.isfinite(xyz)):
            continue
        xyzs.append(xyz)
        kept.append(r)

    if not xyzs:
        raise ValueError(f"No tcp_after.{arm} xyz found for episode={episode} in {log_path}")

    P = np.stack(xyzs, axis=0)  # [T,3]
    dP = np.diff(P, axis=0)
    step_disp = np.linalg.norm(dP, axis=1)  # [T-1]

    # top jump steps
    step_disp_top: List[Dict[str, Any]] = []
    if step_disp.size:
        idx = np.argsort(step_disp)[-topk:][::-1]
        for i in idx:
            step_disp_top.append(
                {
                    "step": int(kept[i + 1].get("step", i + 1)),
                    "plan_id": int(kept[i + 1].get("plan_id", -1)),
                    "disp_m": float(step_disp[i]),
                    "pos": [float(x) for x in P[i + 1].tolist()],
                }
            )

    # rewind-like teleports: large jump AND land close to a past pose
    rewind_examples: List[Dict[str, Any]] = []
    rewind_count = 0
    if step_disp.size:
        for t in range(1, P.shape[0]):
            disp = float(np.linalg.norm(P[t] - P[t - 1]))
            if disp < float(teleport_thresh):
                continue
            start = max(0, t - int(rewind_window_max))
            end = max(0, t - int(rewind_window_min))
            if end <= start:
                continue
            past = P[start:end]
            d = np.linalg.norm(past - P[t], axis=1)
            min_d = float(d.min()) if d.size else float("inf")
            if min_d < float(rewind_return_thresh):
                rewind_count += 1
                if len(rewind_examples) < topk:
                    argmin = int(d.argmin())
                    t_past = start + argmin
                    rewind_examples.append(
                        {
                            "step": int(kept[t].get("step", t)),
                            "plan_id": int(kept[t].get("plan_id", -1)),
                            "disp_m": disp,
                            "return_dist_m": min_d,
                            "matched_past_step": int(kept[t_past].get("step", t_past)),
                        }
                    )

    summary = TeleportSummary(
        log=str(log_path),
        name=str(name),
        arm=str(arm),
        episode=int(episode),
        n_steps=int(P.shape[0]),
        step_disp_m=_percentiles(step_disp),
        step_disp_top=step_disp_top,
        rewind_teleports=int(rewind_count),
        rewind_examples=rewind_examples,
    )
    return summary, P


def _plot(out_path: Path, name_a: str, P_a: np.ndarray, name_b: str, P_b: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_a = np.arange(P_a.shape[0])
    t_b = np.arange(P_b.shape[0])

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    for k, axis_name in enumerate(["x", "y", "z"]):
        axes[k].plot(t_a, P_a[:, k], label=f"{name_a}-{axis_name}", linewidth=1.5)
        axes[k].plot(t_b, P_b[:, k], label=f"{name_b}-{axis_name}", linewidth=1.5, alpha=0.8)
        axes[k].set_ylabel(axis_name)
        axes[k].grid(True, alpha=0.2)
        axes[k].legend(loc="best", fontsize=9)

    d_a = np.linalg.norm(np.diff(P_a, axis=0), axis=1)
    d_b = np.linalg.norm(np.diff(P_b, axis=0), axis=1)
    axes[3].plot(np.arange(d_a.shape[0]), d_a, label=f"{name_a}-step_disp(m)", linewidth=1.5)
    axes[3].plot(np.arange(d_b.shape[0]), d_b, label=f"{name_b}-step_disp(m)", linewidth=1.5, alpha=0.8)
    axes[3].set_ylabel("step disp (m)")
    axes[3].set_xlabel("step")
    axes[3].grid(True, alpha=0.2)
    axes[3].legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log_a", type=str, required=True)
    p.add_argument("--name_a", type=str, default="A")
    p.add_argument("--log_b", type=str, required=True)
    p.add_argument("--name_b", type=str, default="B")
    p.add_argument("--arm", type=str, default="right", choices=["left", "right"])
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--teleport_thresh", type=float, default=0.05, help="单步 TCP 位移超过该阈值视为闪现候选（米）")
    p.add_argument("--rewind_return_thresh", type=float, default=0.02, help="若落点接近过去位置，小于该阈值视为回跳（米）")
    p.add_argument("--rewind_window_min", type=int, default=10, help="回跳匹配窗口下界（步）")
    p.add_argument("--rewind_window_max", type=int, default=80, help="回跳匹配窗口上界（步）")
    args = p.parse_args()

    log_a = Path(args.log_a).expanduser()
    log_b = Path(args.log_b).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_a = _read_jsonl(log_a)
    rows_b = _read_jsonl(log_b)

    s_a, P_a = analyze_log(
        rows_a,
        log_path=log_a,
        name=args.name_a,
        arm=args.arm,
        episode=int(args.episode),
        topk=int(args.topk),
        teleport_thresh=float(args.teleport_thresh),
        rewind_return_thresh=float(args.rewind_return_thresh),
        rewind_window_min=int(args.rewind_window_min),
        rewind_window_max=int(args.rewind_window_max),
    )
    s_b, P_b = analyze_log(
        rows_b,
        log_path=log_b,
        name=args.name_b,
        arm=args.arm,
        episode=int(args.episode),
        topk=int(args.topk),
        teleport_thresh=float(args.teleport_thresh),
        rewind_return_thresh=float(args.rewind_return_thresh),
        rewind_window_min=int(args.rewind_window_min),
        rewind_window_max=int(args.rewind_window_max),
    )

    # 写出 summary
    out_summary = out_dir / f"tcp_compare_{args.arm}_ep{args.episode}.json"
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump({"a": s_a.__dict__, "b": s_b.__dict__}, f, ensure_ascii=False, indent=2)

    # 写图
    out_png = out_dir / f"tcp_compare_{args.arm}_ep{args.episode}.png"
    _plot(out_png, s_a.name, P_a, s_b.name, P_b)

    print("== TCP Trajectory Compare ==")
    print("summary:", out_summary)
    print("plot:   ", out_png)
    print("")
    print(f"[{s_a.name}] n_steps={s_a.n_steps}")
    print("  step_disp_m:", s_a.step_disp_m)
    print("  rewind_teleports:", s_a.rewind_teleports)
    print("  top_step_disp:", s_a.step_disp_top[: min(5, len(s_a.step_disp_top))])
    print("")
    print(f"[{s_b.name}] n_steps={s_b.n_steps}")
    print("  step_disp_m:", s_b.step_disp_m)
    print("  rewind_teleports:", s_b.rewind_teleports)
    print("  top_step_disp:", s_b.step_disp_top[: min(5, len(s_b.step_disp_top))])


if __name__ == "__main__":
    main()

