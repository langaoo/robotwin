"""
DP2DP3 Policy Package for RoBoTwin

DP2DP3 是一个基于 RGB 图像的 Diffusion Policy，通过蒸馏学习将 RGB 特征对齐到点云特征空间。

架构：
- RGB 图像 → 4 个视觉模型 (CroCo/VGGT/DINOv3/DA3)
- 对齐编码器 (RGB2PC) → 统一特征空间
- Diffusion Policy Head → 动作输出

使用方法：
1. 训练对齐编码器：
   cd features_model
   python tools/train_rgb2pc_distill.py --config configs/alignment/train_rgb2pc_distill_default.yaml

2. 训练动作头：
   cd features_model
   python tools/train_online_from_config.py --config configs/head/train_online_batch_extract.yaml

3. 推理（通过 RoBoTwin）：
   cd /home/gl/RoboTwin
   bash policy/DP2DP3/eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id>
"""

from .deploy_policy import get_model, encode_obs, DP2DP3Model, eval, reset_model

__all__ = ['get_model', 'encode_obs', 'DP2DP3Model', 'eval', 'reset_model']
__version__ = '1.0.0'
