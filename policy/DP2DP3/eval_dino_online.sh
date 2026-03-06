#!/bin/bash
# 消融实验: 纯 DINOv3 评估脚本 (无 DA3)
#
# 用法:
#   bash eval_dino_online.sh <task> <task_config> <ckpt_setting> <N> <seed> <gpu_ids> [ckpt_num] [n_exec] [test_num]
#
# 示例:
#   bash eval_dino_online.sh beat_block_hammer demo_clean demo_clean 50 0 "0" 600 6 100

task_name=${1:-beat_block_hammer}
task_config=${2:-demo_clean}
ckpt_setting=${3:-demo_clean}
expert_data_num=${4:-50}
seed=${5:-0}
gpu_ids=${6:-"0"}
checkpoint_num=${7:-600}
n_action_exec=${8:-6}
eval_test_num=${9:-100}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_dino_online_policy.yml"

CKPT_DIR="${SCRIPT_DIR}/checkpoints_dino_online/${task_name}-${ckpt_setting}-${expert_data_num}-${seed}"
if [ -d "${CKPT_DIR}" ]; then
    export DP2DP3_DINO_ONLINE_CKPT_DIR="${CKPT_DIR}"
    echo "[DinoOnline] CKPT dir: ${CKPT_DIR}"
else
    echo "[DinoOnline] CKPT dir not found: ${CKPT_DIR}"
    echo "[DinoOnline] 将由 deploy 脚本自动搜索 checkpoints_dino_online/"
fi

echo "[DinoOnline] Task: ${task_name}  GPU: ${gpu_ids}  Ckpt: ${checkpoint_num}"

bash "${SCRIPT_DIR}/eval.sh" \
    "${task_name}" \
    "${task_config}" \
    "${ckpt_setting}" \
    "${expert_data_num}" \
    "${seed}" \
    "${gpu_ids}" \
    "${checkpoint_num}" \
    "${n_action_exec}" \
    "${eval_test_num}"
