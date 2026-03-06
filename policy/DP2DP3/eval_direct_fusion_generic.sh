#!/bin/bash
# 通用 N 模型直融 Eval 脚本
#
# 用法:
#   bash eval_direct_fusion_generic.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_ids> [checkpoint_num] [n_action_exec] [eval_test_num] [ckpt_suffix]
#
# ckpt_suffix: checkpoint 目录后缀，例如 "croco_dinov3-proprio", "3model_no_vggt-proprio", "4model-proprio"
#              对应 checkpoints_direct_fusion_ws1/<task>-<config>-<N>-<seed>-<suffix>/
#
# 示例:
#   bash eval_direct_fusion_generic.sh lift_pot demo_clean demo_clean 50 0 "0,1" 600 6 100 croco_dinov3-proprio
#   bash eval_direct_fusion_generic.sh lift_pot demo_clean demo_clean 50 0 "0,1" 600 6 100 3model_no_vggt-proprio
#   bash eval_direct_fusion_generic.sh lift_pot demo_clean demo_clean 50 0 "0,1" 600 6 100 4model-proprio

policy_name=DP2DP3.deploy_direct_fusion_policy
task_name=${1:-lift_pot}
task_config=${2:-demo_clean}
ckpt_setting=${3:-demo_clean}
expert_data_num=${4:-50}
seed=${5:-0}
gpu_ids=${6:-"0,1"}
checkpoint_num=${7:-600}
n_action_exec=${8:-6}
eval_test_num=${9:-100}
ckpt_suffix=${10:-"croco_dinov3-proprio"}

# 设置 checkpoint 根目录
export DP2DP3_DIRECT_FUSION_CKPT_ROOTS="/home/gl/RoboTwin/policy/DP2DP3/checkpoints_direct_fusion_ws1:/home/gl/RoboTwin/policy/DP2DP3/features_model/checkpoints_direct_fusion_ws1"

export CUDA_VISIBLE_DEVICES=${gpu_ids}
export HYDRA_FULL_ERROR=1

echo -e "\033[33m[DirectFusion-Generic] Task: ${task_name}\033[0m"
echo -e "\033[33m[DirectFusion-Generic] GPU(s): ${gpu_ids}\033[0m"
echo -e "\033[33m[DirectFusion-Generic] Checkpoint suffix: ${ckpt_suffix}\033[0m"
echo -e "\033[33m[DirectFusion-Generic] Checkpoint num: ${checkpoint_num}\033[0m"
echo -e "\033[33m[DirectFusion-Generic] n_action_exec: ${n_action_exec}\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

# 构建 ckpt_setting: 把 suffix 附加上去，让 deploy 找到正确目录
# deploy 会搜索: <task>-<ckpt_setting>-<N>-<seed>-<suffix>
# 所以实际 ckpt_setting 传 "demo_clean" 即可，目录名由 suffix 决定
# ★ 但是 deploy 搜索逻辑按 <task>-<ckpt_setting>-<N>-<seed>[-suffix] 枚举
# 为了通用性，我们直接指定 checkpoint 根目录和 ckpt_setting 使其匹配
# 
# deploy_direct_fusion_policy.py 搜索:
#   checkpoints_dir / "<task>-<ckpt_setting>-<N>-<seed>-3model-proprio"
#   checkpoints_dir / "<task>-<ckpt_setting>-<N>-<seed>-proprio"
#   checkpoints_dir / "<task>-<ckpt_setting>-<N>-<seed>"
#
# 通用方案: 直接通过环境变量传递完整 checkpoint 目录
CKPT_DIR="/home/gl/RoboTwin/policy/DP2DP3/checkpoints_direct_fusion_ws1/${task_name}-${ckpt_setting}-${expert_data_num}-${seed}-${ckpt_suffix}"
if [ ! -d "${CKPT_DIR}" ]; then
    # 也检查 features_model 下
    CKPT_DIR="/home/gl/RoboTwin/policy/DP2DP3/features_model/checkpoints_direct_fusion_ws1/${task_name}-${ckpt_setting}-${expert_data_num}-${seed}-${ckpt_suffix}"
fi

if [ ! -d "${CKPT_DIR}" ]; then
    echo -e "\033[31m[ERROR] Checkpoint dir not found: ${CKPT_DIR}\033[0m"
    exit 1
fi
echo -e "\033[33m[DirectFusion-Generic] Checkpoint dir: ${CKPT_DIR}\033[0m"

# ★ 通过环境变量直接指定 checkpoint 目录（绕过 deploy 的自动搜索）
export DP2DP3_DIRECT_FUSION_CKPT_DIR="${CKPT_DIR}"

# 使用 proprio deploy yml
DEPLOY_YML=policy/DP2DP3/deploy_direct_fusion_proprio_policy.yml
export DP2DP3_DEPLOY_CONFIG=${DEPLOY_YML}

cmd="python script/eval_policy.py \
  --config ${DEPLOY_YML} \
  --overrides \
  --task_name ${task_name} \
  --task_config ${task_config} \
  --ckpt_setting ${ckpt_setting} \
  --expert_data_num ${expert_data_num} \
  --seed ${seed} \
  --policy_name ${policy_name} \
  --checkpoint_num ${checkpoint_num} \
  --gpu_id 0 \
  --n_action_exec ${n_action_exec}"

if [ -n "${eval_test_num}" ]; then
    cmd="${cmd} --eval_test_num ${eval_test_num}"
fi

echo -e "\033[36m${cmd}\033[0m"
eval ${cmd}
