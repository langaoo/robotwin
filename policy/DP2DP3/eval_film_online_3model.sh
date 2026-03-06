#!/bin/bash
# 在线 3 模型 DA3-FiLM 评估脚本 (DINOv3 + CroCo + DA3)
#
# 用法:
#   bash eval_film_online_3model.sh <task> <task_config> <ckpt_setting> <N> <seed> <gpu_ids> [ckpt_num] [n_exec] [test_num]
#
# 示例:
#   bash eval_film_online_3model.sh lift_pot demo_clean demo_clean 50 0 "0" 600 6 100

task_name=${1:-lift_pot}
task_config=${2:-demo_clean}
ckpt_setting=${3:-demo_clean}
expert_data_num=${4:-50}
seed=${5:-0}
gpu_ids=${6:-"0"}
checkpoint_num=${7:-600}
n_action_exec=${8:-6}
eval_test_num=${9:-100}

# 脚本所在目录 (DP2DP3/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# deploy yml -> 3 模型 FiLM 在线版
export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_film_online_3model_policy.yml"

# Checkpoint 目录
CKPT_DIR="${SCRIPT_DIR}/checkpoints_film_online_3model/${task_name}-${ckpt_setting}-${expert_data_num}-${seed}"
if [ -d "${CKPT_DIR}" ]; then
    export DP2DP3_FILM_ONLINE_3MODEL_CKPT_DIR="${CKPT_DIR}"
    echo -e "\033[33m[Film3Model] CKPT dir: ${CKPT_DIR}\033[0m"
else
    echo -e "\033[33m[Film3Model] CKPT dir not found: ${CKPT_DIR}\033[0m"
    echo -e "\033[33m[Film3Model] 将由 deploy 脚本自动搜索 checkpoints_film_online_3model/\033[0m"
fi

echo -e "\033[33m[Film3Model] Task: ${task_name}  GPU: ${gpu_ids}  Ckpt: ${checkpoint_num}\033[0m"

# 调用标准 eval.sh
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
