#!/bin/bash
# DP2DP3 Direct Fusion 3Model + Proprio Policy Evaluation Script
# 
# 3模型直接融合 (CroCo + VGGT + DINOv3，去掉DA3) + proprio
# 用于与 depth_guided (3RGB + DA3深度) 的对比基线实验
#
# 使用方法:
#   bash eval_direct_fusion_3model.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> [checkpoint_num] [n_action_exec] [eval_test_num]
#
# 示例:
#   bash eval_direct_fusion_3model.sh lift_pot demo_clean demo_clean 50 0 0 600 6 100

# == 基本参数 ==
policy_name=DP2DP3.deploy_direct_fusion_policy
task_name=${1:-lift_pot}
task_config=${2:-demo_clean}
ckpt_setting=${3:-demo_clean}
expert_data_num=${4:-50}
seed=${5:-0}
gpu_ids=${6:-1}
checkpoint_num=${7:-best}
n_action_exec=${8:-6}
eval_test_num=${9:-}

# 固定使用 3model proprio yml
DEPLOY_YML=policy/DP2DP3/deploy_direct_fusion_3model_proprio_policy.yml

# 设置 GPU
export CUDA_VISIBLE_DEVICES=${gpu_ids}
primary_gpu_id=${gpu_ids%%,*}
export HYDRA_FULL_ERROR=1
echo -e "\033[36m[DP2DP3-DirectFusion-3Model] Policy Name: ${policy_name}\033[0m"
echo -e "\033[36m[DP2DP3-DirectFusion-3Model] Task: ${task_name}\033[0m"
echo -e "\033[36m[DP2DP3-DirectFusion-3Model] Expert Data Num: ${expert_data_num}\033[0m"
echo -e "\033[36m[DP2DP3-DirectFusion-3Model] Using Physical GPU(s): ${gpu_ids}\033[0m"
echo -e "\033[36m[DP2DP3-DirectFusion-3Model] Checkpoint: ${checkpoint_num}\033[0m"
echo -e "\033[36m[DP2DP3-DirectFusion-3Model] n_action_exec: ${n_action_exec}\033[0m"
echo -e "\033[36m[DP2DP3-DirectFusion-3Model] Seed: ${seed}\033[0m"
echo -e "\033[36m[DP2DP3-DirectFusion-3Model] Models: CroCo + VGGT + DINOv3 (no DA3)\033[0m"

# 切换到 RoboTwin 根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

# Python 解释器
PYTHON_BIN=${PYTHON_BIN:-/home/gl/miniconda3/envs/RoboTwin/bin/python}
if ! command -v ${PYTHON_BIN} >/dev/null 2>&1; then
    PYTHON_BIN=python
fi

# 运行评估
PYTHONWARNINGS=ignore::UserWarning \
${PYTHON_BIN} script/eval_policy.py --config ${DEPLOY_YML} \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --checkpoint_num ${checkpoint_num} \
    --gpu_id 0 \
    --n_action_exec ${n_action_exec} \
    --gpu_ids "[${gpu_ids}]" \
    ${eval_test_num:+--eval_test_num ${eval_test_num}}
