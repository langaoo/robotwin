#!/bin/bash
# DP2DP3 Dual-Stream Legacy Policy Evaluation Script
# 用于 train_offline_head1.py 训练的 dual-stream head（无 token_gate）

policy_name=DP2DP3.deploy_dual_stream_legacy_policy
task_name=${1:-lift_pot}
task_config=${2:-demo_clean}
ckpt_setting=${3:-demo_clean_dual_stream}
expert_data_num=${4:-50}
seed=${5:-0}
gpu_ids=${6:-1}
checkpoint_num=${7:-best}
n_action_exec=${8:-4}
eval_test_num=${9:-}

export CUDA_VISIBLE_DEVICES=${gpu_ids}
primary_gpu_id=${gpu_ids%%,*}
export HYDRA_FULL_ERROR=1

echo -e "\033[33m[DP2DP3-LegacyDualStream] Policy Name: ${policy_name}\033[0m"
echo -e "\033[33m[DP2DP3-LegacyDualStream] Task: ${task_name}\033[0m"
echo -e "\033[33m[DP2DP3-LegacyDualStream] Expert Data Num: ${expert_data_num}\033[0m"
echo -e "\033[33m[DP2DP3-LegacyDualStream] Using Physical GPU(s): ${gpu_ids}\033[0m"
echo -e "\033[33m[DP2DP3-LegacyDualStream] Checkpoint: ${checkpoint_num}\033[0m"
echo -e "\033[33m[DP2DP3-LegacyDualStream] n_action_exec: ${n_action_exec}\033[0m"
echo -e "\033[33m[DP2DP3-LegacyDualStream] Seed: ${seed}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/DP2DP3/deploy_dual_stream_legacy_policy.yml \
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
    ${eval_test_num:+--eval_test_num ${eval_test_num}}
