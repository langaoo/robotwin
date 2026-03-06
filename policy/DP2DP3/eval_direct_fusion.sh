#!/bin/bash
# DP2DP3 Direct Fusion Policy Evaluation Script for RoBoTwin
# 
# 使用方法:
#   bash eval_direct_fusion.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> [checkpoint_num] [n_action_exec] [eval_test_num] [use_proprio]
#
# use_proprio: 1=使用本体感知(proprio)版本, 0=不使用(默认1)
#
# 示例（有 proprio）:
#   bash eval_direct_fusion.sh lift_pot demo_clean demo_clean 50 0 0 600 6 100 1
# 示例（无 proprio，旧行为）:
#   bash eval_direct_fusion.sh lift_pot demo_clean demo_clean 50 0 0 600 6 100 0

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
use_proprio=${10:-1}   # 默认使用 proprio 版本

# 根据 use_proprio 选择 yml
if [ "${use_proprio}" = "1" ]; then
    DEPLOY_YML=policy/DP2DP3/deploy_direct_fusion_proprio_policy.yml
else
    DEPLOY_YML=policy/DP2DP3/deploy_direct_fusion_policy.yml
fi

# 设置 GPU
# 支持单卡 (例如 "1") 或双卡/多卡 (例如 "0,1")
export CUDA_VISIBLE_DEVICES=${gpu_ids}
primary_gpu_id=${gpu_ids%%,*}
export HYDRA_FULL_ERROR=1
echo -e "\033[33m[DP2DP3-DirectFusion] Policy Name: ${policy_name}\033[0m"
echo -e "\033[33m[DP2DP3-DirectFusion] Task: ${task_name}\033[0m"
echo -e "\033[33m[DP2DP3-DirectFusion] Expert Data Num: ${expert_data_num}\033[0m"
echo -e "\033[33m[DP2DP3-DirectFusion] Using Physical GPU(s): ${gpu_ids}\033[0m"
echo -e "\033[33m[DP2DP3-DirectFusion] Checkpoint: ${checkpoint_num}\033[0m"
echo -e "\033[33m[DP2DP3-DirectFusion] n_action_exec: ${n_action_exec}\033[0m"
echo -e "\033[33m[DP2DP3-DirectFusion] Seed: ${seed}\033[0m"
echo -e "\033[33m[DP2DP3-DirectFusion] use_proprio: ${use_proprio}  →  ${DEPLOY_YML}\033[0m"

# 切换到 RoboTwin 根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

# Python 解释器
PYTHON_BIN=${PYTHON_BIN:-/home/gl/miniconda3/envs/RoboTwin/bin/python}
if ! command -v ${PYTHON_BIN} >/dev/null 2>&1; then
    PYTHON_BIN=python
fi

# 运行评估
# 注意: CUDA_VISIBLE_DEVICES已经映射物理GPU,所以传入gpu_id=0使用映射后的设备
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
