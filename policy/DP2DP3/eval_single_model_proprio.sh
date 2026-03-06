#!/bin/bash
# 单模型 + 本体感知 (proprio) 评估脚本
#
# 用法:
#   bash eval_single_model_proprio.sh <task> <task_config> <ckpt_setting> <expert_data_num> <seed> <gpu_id> <checkpoint_num> <model_name> [n_action_exec] [eval_test_num]
#
# 示例:
#   bash eval_single_model_proprio.sh lift_pot demo_clean demo_clean 50 0 0 600 croco 6 100
#   bash eval_single_model_proprio.sh lift_pot demo_clean demo_clean 50 0 0 600 da3   6 100
#   bash eval_single_model_proprio.sh lift_pot demo_clean demo_clean 50 0 0 600 vggt  6 100
#   bash eval_single_model_proprio.sh lift_pot demo_clean demo_clean 50 0 0 600 dinov3 6 100
#
# 与 eval_single_model.sh 的区别:
#   - 使用 deploy_single_model_proprio_policy.yml (有本体感知)
#   - 自动把 model_name 写入 yml 的 model_name 字段（通过命令行 --overrides 透传）

policy_name=DP2DP3
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
checkpoint_num=${7:-best}
model_name=${8:-dinov3}
n_action_exec=${9:-6}
eval_test_num=${10:-}

echo "============================================"
echo "[SingleModel+Proprio] 单模型本体感知评估"
echo "============================================"
echo "Task:            $task_name"
echo "Task Config:     $task_config"
echo "Ckpt Setting:    $ckpt_setting"
echo "Num Demos:       $expert_data_num"
echo "Seed:            $seed"
echo "GPU:             $gpu_id"
echo "Checkpoint:      $checkpoint_num"
echo "Model:           $model_name"
echo "n_action_exec:   $n_action_exec"
echo "eval_test_num:   ${eval_test_num:-default}"
echo "============================================"

export CUDA_VISIBLE_DEVICES=$gpu_id
export HYDRA_FULL_ERROR=1

# 切换到 RoboTwin 根目录（支持从任意位置调用该脚本）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

# Python 解释器（优先使用 RoboTwin 环境）
PYTHON_BIN=${PYTHON_BIN:-/home/gl/miniconda3/envs/RoboTwin/bin/python}
if ! command -v ${PYTHON_BIN} >/dev/null 2>&1; then
    PYTHON_BIN=python
fi

# 运行评估
# 注意: CUDA_VISIBLE_DEVICES 已将物理 GPU $gpu_id 映射为逻辑 GPU 0
PYTHONWARNINGS=ignore::UserWarning \
${PYTHON_BIN} script/eval_policy.py \
    --config policy/${policy_name}/deploy_single_model_proprio_policy.yml \
    --overrides \
    --task_name        ${task_name} \
    --task_config      ${task_config} \
    --ckpt_setting     ${ckpt_setting} \
    --expert_data_num  ${expert_data_num} \
    --seed             ${seed} \
    --policy_name      ${policy_name} \
    --checkpoint_num   ${checkpoint_num} \
    --gpu_id           0 \
    --n_action_exec    ${n_action_exec} \
    --model_name       ${model_name} \
    ${eval_test_num:+--eval_test_num ${eval_test_num}}
