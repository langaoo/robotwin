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

# 切换到 RoboTwin 根目录
cd ../..

# 运行评估
# 注意: CUDA_VISIBLE_DEVICES已经映射物理GPU,所以传入gpu_id=0使用映射后的设备
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
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
