#!/bin/bash
# 在线 2 模型 DA3-FiLM v2 评估脚本 (DINOv3 + DA3 + AttentionPooling)
#
# 与 eval_film_online.sh 的区别:
#   - 使用 deploy_film_online_v2_policy.yml (v2 encoder)
#   - CKPT 目录为 checkpoints_film_online_v2/
#   - 环境变量 DP2DP3_FILM_ONLINE_V2_CKPT_DIR (v2 专用)
#
# 用法:
#   bash eval_film_online_v2.sh <task> <task_config> <ckpt_setting> <N> <seed> <gpu_ids> [ckpt_num] [n_exec] [test_num]
#
# 示例:
#   bash eval_film_online_v2.sh beat_block_hammer demo_clean demo_clean 50 0 "0" 600 6 30

task_name=${1:-beat_block_hammer}
task_config=${2:-demo_clean}
ckpt_setting=${3:-demo_clean}
expert_data_num=${4:-50}
seed=${5:-0}
gpu_ids=${6:-"0"}
checkpoint_num=${7:-600}
n_action_exec=${8:-6}
eval_test_num=${9:-30}

# 脚本所在目录 (DP2DP3/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# deploy yml -> v2 FiLM (AttentionPooling)
export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_film_online_v2_policy.yml"

# Checkpoint 目录 (v2)
CKPT_DIR="${SCRIPT_DIR}/checkpoints_film_online_v2/${task_name}-${ckpt_setting}-${expert_data_num}-${seed}"
if [ -d "${CKPT_DIR}" ]; then
    export DP2DP3_FILM_ONLINE_V2_CKPT_DIR="${CKPT_DIR}"
    echo -e "[Film2ModelV2] CKPT dir: ${CKPT_DIR}"
else
    echo -e "[Film2ModelV2] CKPT dir not found: ${CKPT_DIR}"
    echo -e "[Film2ModelV2] 将由 deploy 脚本自动搜索 checkpoints_film_online_v2/"
fi

echo -e "[Film2ModelV2] Task: ${task_name}  GPU: ${gpu_ids}  Ckpt: ${checkpoint_num}"

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
