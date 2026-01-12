import argparse
import os

import numpy as np

from process_rgb_pcd import save_ply


def load_ply_xyz(ply_path):
    """简易 PLY 读取，只解析 xyz（忽略颜色/其他属性）。"""
    with open(ply_path, "r") as f:
        header_ended = False
        points = []
        for line in f:
            line = line.strip()
            if not header_ended:
                if line == "end_header":
                    header_ended = True
                continue
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            points.append([x, y, z])
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def densify_points(points, factor=4, noise_std=0.002):
    """对点云做简单上采样: 每个点周围加 (factor-1) 个带小高斯噪声的点。

    :param points: (N, 3) 或 (N, C) 的点云
    :param factor: 上采样倍数，总点数约为 N * factor
    :param noise_std: 高斯噪声标准差
    :return: (N * factor, C) 的新点云
    """
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.size == 0:
        return points

    N, C = points.shape
    if factor <= 1:
        return points

    # 只对 xyz 加噪声，其他维度（例如颜色）保持不变
    base_xyz = points[:, :3]
    others = points[:, 3:] if C > 3 else None

    # 复制 points (factor 次)，然后对 xyz 部分添加噪声
    base_xyz_tiled = np.tile(base_xyz, (factor, 1))
    noise = np.random.normal(scale=noise_std, size=base_xyz_tiled.shape)
    new_xyz = base_xyz_tiled + noise

    if others is not None:
        others_tiled = np.tile(others, (factor, 1))
        new_points = np.concatenate([new_xyz, others_tiled], axis=1)
    else:
        new_points = new_xyz

    return new_points


def process_folder(input_root, output_root=None, factor=4, noise_std=0.002):
    """对一个目录下所有 ply 文件做点云加密 (densify)。

    input_root: 原始 PC/PCD 目录，例如 policy/DP3/rgbpc_dataset/PC/beat_block_hammer-demo_clean-50
    output_root: 输出目录；如果为 None，则覆盖原始 ply。
    """
    if output_root is None:
        output_root = input_root

    for root, dirs, files in os.walk(input_root):
        for fname in files:
            if not fname.endswith(".ply"):
                continue
            in_path = os.path.join(root, fname)

            # 计算输出路径
            if output_root == input_root:
                out_path = in_path
            else:
                rel = os.path.relpath(in_path, input_root)
                out_path = os.path.join(output_root, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

            print(f"densify: {in_path} -> {out_path}")
            pts = load_ply_xyz(in_path)
            dense_pts = densify_points(pts, factor=factor, noise_std=noise_std)
            save_ply(dense_pts, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对已有 PLY 点云做简单上采样，使其更密集")
    parser.add_argument("input_root", type=str, help="输入点云根目录 (PC 或 PCD 路径)")
    parser.add_argument("--output_root", type=str, default=None, help="输出根目录，若不填则就地覆盖")
    parser.add_argument("--factor", type=int, default=4, help="上采样倍数 (每个点扩展为 factor 个)")
    parser.add_argument("--noise_std", type=float, default=0.002, help="高斯噪声标准差，用于生成邻域点")
    args = parser.parse_args()

    process_folder(args.input_root, args.output_root, factor=args.factor, noise_std=args.noise_std)
