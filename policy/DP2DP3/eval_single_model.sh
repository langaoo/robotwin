#!/bin/bash
# 单模型评估脚本
# 用法: bash eval_single_model.sh <task> <train_setting> <eval_setting> <num_demos> <seed> <gpu_id> <checkpoint_num> <model_name>

policy_name=DP2DP3  # 单模型评估输出独立目录
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
checkpoint_num=${7:-best}
model_name=${8:-dinov3}  # 默认dinov3
n_action_exec=${9:-8}

echo "============================================"
echo "单模型评估 - ${model_name}"
echo "============================================"
echo "Task: $task_name"
echo "Train Setting: $ckpt_setting"
echo "Eval Setting: $task_config"
echo "Num Demos: $expert_data_num"
echo "Seed: $seed"
echo "GPU: $gpu_id"
echo "Checkpoint: $checkpoint_num"
echo "Model: $model_name"
echo "============================================"

export CUDA_VISIBLE_DEVICES=$gpu_id
export HYDRA_FULL_ERROR=1

# 切换到 RoboTwin 根目录
cd ../..

# 运行评估
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_single_model_policy.yml \
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
    --model_name ${model_name}
