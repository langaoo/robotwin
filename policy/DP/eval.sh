#!/bin/bash

# == keep unchanged ==
policy_name=DP
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
checkpoint_num=${7:-600}  # 默认 600
DEBUG=False
shift 7 2>/dev/null || true
PASSTHROUGH_ARGS=("$@")

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v ${PYTHON_BIN} >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    fi
fi

PYTHONWARNINGS=ignore::UserWarning \
${PYTHON_BIN} script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --checkpoint_num ${checkpoint_num} \
    --seed ${seed} \
    "${PASSTHROUGH_ARGS[@]}"