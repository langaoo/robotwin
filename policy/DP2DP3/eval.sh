#!/bin/bash
# DP2DP3 Policy Evaluation Script for RoBoTwin
# 
# 使用方法（与 DP/DP3 保持一致）:
#   bash eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> [checkpoint_num]
#
# 示例:
#   bash eval.sh beat_block_hammer demo_randomized demo_randomized 20 0 1 50
#   bash eval.sh beat_block_hammer demo_randomized demo_randomized 20 0 1 best

# == 基本参数 ==
policy_name=DP2DP3
task_name=${1:-beat_block_hammer}           # 任务名称
task_config=${2:-demo_randomized}            # 任务配置
ckpt_setting=${3:-demo_randomized}           # checkpoint 设置
expert_data_num=${4:-20}                     # 专家数据数量
seed=${5:-0}                                 # 随机种子
gpu_ids=${6:-1}                              # GPU IDs (e.g. "0" or "0,1")
checkpoint_num=${7:-50}                      # checkpoint 编号（默认 50）
n_action_exec=${8:-4}                        # Receding Horizon执行步数
eval_test_num=${9:-5}
shift 9 2>/dev/null || true   # 移除前9个位置参数，剩余的作为透传参数
PASSTHROUGH_ARGS=("$@")       # 收集剩余参数 (e.g. --force_gate 0.0)

# 设置 GPU
export CUDA_VISIBLE_DEVICES=${gpu_ids}
primary_gpu_id=${gpu_ids%%,*}
export HYDRA_FULL_ERROR=1
echo -e "\033[33m[DP2DP3] Policy Name: ${policy_name}\033[0m"
echo -e "\033[33m[DP2DP3] Task: ${task_name}\033[0m"
echo -e "\033[33m[DP2DP3] Expert Data Num: ${expert_data_num}\033[0m"
echo -e "\033[33m[DP2DP3] Using Physical GPU(s): ${gpu_ids}\033[0m"
echo -e "\033[33m[DP2DP3] Checkpoint: ${checkpoint_num}\033[0m"
echo -e "\033[33m[DP2DP3] n_action_exec: ${n_action_exec}\033[0m"
echo -e "\033[33m[DP2DP3] Seed: ${seed}\033[0m"

# 切换到 RoboTwin 根目录（兼容从任意位置执行该脚本）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

# Python 解释器选择（默认使用环境里的 python；若不存在则回退到 python3）
PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v ${PYTHON_BIN} >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    fi
fi

# 可选：动作/TCP 日志（JSONL）。不设置则与 baseline 行为完全一致。
# 用法示例：
#   export DP2DP3_ACTION_LOG_PATH=policy/DP2DP3/logs/action_logs/eval100_ckpt600_k4_plan4.jsonl
#   export DP2DP3_EVAL_VIDEO_LOG=0   # 跑 100 episodes 时建议关掉视频，避免磁盘爆炸
EXTRA_ARGS=()
if [[ -n "${DP2DP3_ACTION_LOG_PATH}" ]]; then
    EXTRA_ARGS+=(--action_log_path "${DP2DP3_ACTION_LOG_PATH}")
fi
if [[ -n "${DP2DP3_EVAL_VIDEO_LOG}" ]]; then
    EXTRA_ARGS+=(--eval_video_log "${DP2DP3_EVAL_VIDEO_LOG}")
fi
if [[ -n "${DP2DP3_EVAL_START_SEED}" ]]; then
    EXTRA_ARGS+=(--eval_start_seed "${DP2DP3_EVAL_START_SEED}")
fi
if [[ -n "${DP2DP3_EVAL_START_ID}" ]]; then
    EXTRA_ARGS+=(--eval_start_id "${DP2DP3_EVAL_START_ID}")
fi

# 运行评估
# 注意: CUDA_VISIBLE_DEVICES已经映射物理GPU,所以传入gpu_id=0使用映射后的设备
DEPLOY_CONFIG=${DP2DP3_DEPLOY_CONFIG:-policy/$policy_name/deploy_policy.yml}
PYTHONWARNINGS=ignore::UserWarning \
${PYTHON_BIN} script/eval_policy.py --config ${DEPLOY_CONFIG} \
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
    ${eval_test_num:+--eval_test_num ${eval_test_num}} \
    "${EXTRA_ARGS[@]}" \
    "${PASSTHROUGH_ARGS[@]}"
