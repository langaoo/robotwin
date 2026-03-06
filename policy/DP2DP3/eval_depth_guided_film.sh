#!/bin/bash
# DA3-FiLM Depth-Guided Fusion Eval 脚本
#
# 原理: 设置 DP2DP3_DEPLOY_CONFIG 指向 depth_guided yml，
#       设置 DP2DP3_DEPTH_GUIDED_CKPT_DIR 指向 FiLM checkpoint 目录，
#       然后直接调用标准 eval.sh（接口完全复用）
#
# 用法:
#   bash eval_depth_guided_film.sh <task> <task_config> <ckpt_setting> <N> <seed> <gpu_ids> [ckpt_num] [n_exec] [test_num]
#
# 示例:
#   bash eval_depth_guided_film.sh lift_pot demo_clean demo_clean 50 0 "0,1" 600 6 100

task_name=${1:-lift_pot}
task_config=${2:-demo_clean}
ckpt_setting=${3:-demo_clean}
expert_data_num=${4:-50}
seed=${5:-0}
gpu_ids=${6:-"0,1"}
checkpoint_num=${7:-600}
n_action_exec=${8:-6}
eval_test_num=${9:-100}

# 脚本所在目录（DP2DP3/）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# deploy yml —— 复用 cross-attn 的 yml，deploy 脚本会从 ckpt 自动识别 FiLM
export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_depth_guided_policy.yml"

# FiLM checkpoint 目录（优先级高于 deploy 脚本的自动搜索）
FILM_CKPT_DIR="${SCRIPT_DIR}/checkpoints_depth_guided_film/${task_name}-${ckpt_setting}-${expert_data_num}-${seed}"
if [ -d "${FILM_CKPT_DIR}" ]; then
    export DP2DP3_DEPTH_GUIDED_CKPT_DIR="${FILM_CKPT_DIR}"
    echo -e "[33m[DA3Film] CKPT dir: ${FILM_CKPT_DIR}[0m"
else
    echo -e "[33m[DA3Film] CKPT dir not found: ${FILM_CKPT_DIR}[0m"
    echo -e "[33m[DA3Film] 将由 deploy 脚本自动搜索 checkpoints_depth_guided_film/ 和 checkpoints_depth_guided/[0m"
fi

echo -e "[33m[DA3Film] Task: ${task_name}  GPU: ${gpu_ids}  Ckpt: ${checkpoint_num}[0m"

# 直接调用标准 eval.sh，接口完全一致
bash "${SCRIPT_DIR}/eval.sh"     "${task_name}"     "${task_config}"     "${ckpt_setting}"     "${expert_data_num}"     "${seed}"     "${gpu_ids}"     "${checkpoint_num}"     "${n_action_exec}"     "${eval_test_num}"
