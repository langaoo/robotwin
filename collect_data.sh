#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
gpu_id=${3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v ${PYTHON_BIN} >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  fi
fi

${PYTHON_BIN} script/update_embodiment_config_path.py > /dev/null 2>&1 || true

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
${PYTHON_BIN} script/collect_data.py $task_name $task_config
rm -rf data/${task_name}/${task_config}/.cache
