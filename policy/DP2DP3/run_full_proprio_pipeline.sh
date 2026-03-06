#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gl/RoboTwin"
FEATURES="${ROOT}/policy/DP2DP3/features_model"
LOG_DIR="${ROOT}/policy/DP2DP3/logs/auto_runs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/full_proprio_pipeline_$(date +%Y%m%d_%H%M%S).log"

PY_TRAIN="/home/gl/miniconda3/envs/depth3/bin/python"
PY_EVAL="/home/gl/miniconda3/envs/RoboTwin/bin/python"

exec >> "${MASTER_LOG}" 2>&1

trap 'echo "[ERROR] line ${LINENO}, exit code $?"' ERR
trap 'echo "[EXIT] code $?"' EXIT

_echo_step() {
  echo
  echo "===== $1 ====="
  date
}

echo "[INFO] Master log: ${MASTER_LOG}"

# 1) token_full + proprio
_echo_step "Step1: token_full + proprio (extract)"
cd "${FEATURES}"
echo "[PWD] $(pwd)"
${PY_TRAIN} tools/offline_proprio/extract_offline_features_proprio.py \
  --config configs/head/train_online_batch_extract_dual_stream_tokens_full_proprio.yaml \
  --output_dir "${FEATURES}/data/offline_features_dual_stream_tokens_full_proprio" \
  --overwrite

_echo_step "Step1: token_full + proprio (train)"
${PY_TRAIN} tools/offline_proprio/train_offline_head_proprio.py \
  --config configs/head/train_offline_dual_stream_tokens_full_proprio.yaml

_echo_step "Step1: token_full + proprio (eval 100)"
cd "${ROOT}"
export PYTHON_BIN="${PY_EVAL}"
export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_policy_proprio.yml"
export DP2DP3_EVAL_VIDEO_LOG=0
bash policy/DP2DP3/eval.sh \
  lift_pot demo_clean demo_clean_dual_stream_tokens_full_proprio_shift1 \
  50 0 "0" 600 4 100

# 2) pool_ws1 + proprio
_echo_step "Step2: pool_ws1 + proprio (extract)"
cd "${FEATURES}"
echo "[PWD] $(pwd)"
${PY_TRAIN} tools/offline_proprio/extract_offline_features_proprio.py \
  --config configs/head/train_online_batch_extract_pool_ws1_proprio.yaml \
  --output_dir "${FEATURES}/data/offline_features_pool_ws1_proprio" \
  --overwrite

_echo_step "Step2: pool_ws1 + proprio (train)"
${PY_TRAIN} tools/offline_proprio/train_offline_head_proprio.py \
  --config configs/head/train_offline_pool_ws1_proprio.yaml

_echo_step "Step2: pool_ws1 + proprio (eval 100)"
cd "${ROOT}"
export PYTHON_BIN="${PY_EVAL}"
export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_policy_pool_ws1_proprio.yml"
export DP2DP3_EVAL_VIDEO_LOG=0
bash policy/DP2DP3/eval.sh \
  lift_pot demo_clean demo_clean_pool_ws1_proprio \
  50 0 "0" 600 4 100

# 3) multitask encoder + place_a2b_left + proprio
_echo_step "Step3: place_a2b_left + proprio (extract w/ multitask encoder)"
cd "${FEATURES}"
echo "[PWD] $(pwd)"
${PY_TRAIN} tools/offline_proprio/extract_offline_features_proprio.py \
  --config configs/head/train_online_batch_extract_dual_stream_tokens_full_train.yaml \
  --output_dir "${FEATURES}/data/offline_features_train" \
  --overwrite

_echo_step "Step3: place_a2b_left + proprio (train head)"
${PY_TRAIN} tools/offline_proprio/train_offline_head_proprio.py \
  --config configs/head/train_offline_dual_stream_tokens_full_train.yaml

_echo_step "Step3: place_a2b_left + proprio (eval 100)"
cd "${ROOT}"
export PYTHON_BIN="${PY_EVAL}"
export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_policy_train_proprio.yml"
export DP2DP3_EVAL_VIDEO_LOG=0
bash policy/DP2DP3/eval.sh \
  place_a2b_left demo_clean demo_clean_train_mtenc \
  50 0 "0" 600 4 100

echo
echo "[DONE] All steps finished. Master log: ${MASTER_LOG}"
