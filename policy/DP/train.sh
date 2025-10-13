#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_id=${6}
# 可选: 第7个参数指定 batch_size；不填则默认 128（与配置文件中的原始设定一致）
# 注意: 原先写成 ${7:64} 是子串截取写法，会导致 batch_size 为空，这里修正为 ${7:-128}
batch_size=${7:-128}
epoch_num=${8:-600}  # 训练轮数
isfinetune=${9:-False}  # 是否微调
basemodel=${10:-}  # 微调模型路径
resume_from=${11:-}  # 继续训练的模型路径


head_camera_type=D435

DEBUG=False
save_ckpt=True

alg_name=robot_dp_$action_dim
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[36m(batch size override -> ${batch_size})\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
fi

python train.py --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            setting=${task_config} \
                            expert_data_num=${expert_data_num} \
                            head_camera_type=$head_camera_type \
                            dataloader.batch_size=${batch_size} \
                            val_dataloader.batch_size=${batch_size} \
                            finetune.isfinetune=${isfinetune} \
                            training.num_epochs=${epoch_num} \
                            finetune.resume_from=${resume_from} \
                            finetune.base_model=${basemodel} \
                            # checkpoint.save_ckpt=${save_ckpt}
                            # hydra.run.dir=${run_dir} \