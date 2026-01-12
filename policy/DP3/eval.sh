#!/bin/bash

policy_name=DP3
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5} # both policy and RoboTwin scene
gpu_id=${6}
sampling_method=${7:-"none"}  # 第7个参数：采样方式，默认"none"表示不使用RGBD

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# 只在指定了采样方式时才传递参数
if [ "$sampling_method" != "none" ]; then
    echo -e "\033[36msampling method: ${sampling_method}\033[0m"
    SAMPLING_ARG="--sampling_method ${sampling_method}"
else
    echo -e "\033[36musing original Sapien pointcloud\033[0m"
    SAMPLING_ARG=""
fi

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    ${SAMPLING_ARG}
