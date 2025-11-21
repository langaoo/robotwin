#!/usr/bin/env python3
"""
测试断点续训配置
"""
import torch
import pickle
from pathlib import Path

# 检查checkpoint
ckpt_path = Path("/home/gl/RoboTwin/policy/DP/checkpoints/data/beat_block_hammer-demo_trans-50_h-0/600.ckpt")

if not ckpt_path.exists():
    print(f"❌ Checkpoint不存在: {ckpt_path}")
    exit(1)

print(f"✅ Checkpoint存在: {ckpt_path}")
print(f"   文件大小: {ckpt_path.stat().st_size / 1e9:.2f} GB")

# 加载checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")
epoch = pickle.loads(ckpt['pickles']['epoch'])
global_step = pickle.loads(ckpt['pickles']['global_step'])

print(f"\n📊 Checkpoint信息:")
print(f"   Epoch: {epoch}")
print(f"   Global step: {global_step}")

# 计算训练计划
target_epochs = 619
additional_epochs = 20

print(f"\n📅 训练计划:")
print(f"   当前epoch: {epoch}")
print(f"   目标epoch: {target_epochs}")
print(f"   需要训练: {target_epochs - epoch} 个epoch")
print(f"   期望训练: {additional_epochs} 个epoch")

if target_epochs - epoch == additional_epochs:
    print(f"   ✅ 配置正确！")
else:
    print(f"   ❌ 配置错误！应该设置 num_epochs = {epoch + additional_epochs}")

# 检查保存频率
checkpoint_every = 10
print(f"\n💾 保存计划 (每{checkpoint_every}个epoch):")
save_points = []
for e in range(epoch + 1, target_epochs + 1):
    if e % checkpoint_every == 0:
        save_points.append(e)

print(f"   将在以下epoch保存: {save_points}")
print(f"   预计保存 {len(save_points)} 个checkpoint")
