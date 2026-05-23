#!/bin/bash

# 检查是否传入了需要等待的 PID
    if [ "$#" -eq 0 ]; then
    echo "用法: $0 <PID1> [PID2] [PID3] ..."
    echo "请提供至少一个需要等待的进程 PID。例如: ./wait_and_run.sh 1234 5678"
    exit 1
    fi

echo "开始监控，等待以下进程结束: $@"

# 遍历你传入的所有 PID，并每 10 秒检查一次它们是否还在运行
    for pid in "$@"; do
        while kill -0 "$pid" 2>/dev/null; do
            sleep 300
        done
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] 进程 $pid 已结束。"
    done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 所有指定的进程均已结束，准备开始执行新任务..."

# 激活环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "/home/gl/miniconda3/etc/profile.d/conda.sh" ]; then
  source /home/gl/miniconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] 未找到 conda.sh，请先确认 conda 安装路径。"
  exit 1
fi
conda activate RoboTwin_gl
echo "已激活 RoboTwin 环境"

# 切换目录
cd "${ROOT_DIR}/policy/DP2DP3"

# ========== 独立执行每个评估命令 ==========
# 这里去掉了 &&，改成顺序执行，任何一行报错都不会影响下一行的运行

# echo "--- [1/6] 正在运行: direct_fusion_3model ---"
# bash eval_direct_fusion_3model.sh lift_pot demo_clean demo_clean 50 0 "0,1" 600 6 100

# echo "--- [2/6] 正在运行: tokens_full_global_only_dp_aligned_proprio ---"
# DP2DP3_DEPLOY_CONFIG=policy/DP2DP3/deploy_policy_tokens_full_global_only_dp_aligned_proprio.yml bash eval.sh lift_pot demo_clean demo_clean_tokens_full_global_only_dp_aligned_proprio 50 0 "0,1" 600 6 100

# echo "--- [3/6] 正在运行: depth_guided_policy ---"
# DP2DP3_DEPLOY_CONFIG=policy/DP2DP3/deploy_depth_guided_policy.yml bash eval.sh lift_pot demo_clean demo_clean 50 0 "0,1" 600 6 100

# echo "--- [4/6] 正在运行: single_model_proprio (da3) ---"
# bash eval_single_model_proprio.sh lift_pot demo_clean demo_clean 50 0 0 600 da3  6 100

# echo "--- [5/6] 正在运行: single_model_proprio (croco) ---"
# bash eval_single_model_proprio.sh lift_pot demo_clean demo_clean 50 0 0 600 croco  6 100

# echo "--- [6/6] 正在运行: single_model_proprio (vggt) ---"
# bash eval_single_model_proprio.sh lift_pot demo_clean demo_clean 50 0 0 600 vggt   6 100

# echo "--- [7/6] 正在运行: single_model_proprio (dino) ---"
# bash eval_single_model_proprio.sh beat_block_hammer demo_clean demo_clean 50 0 0 600 dino   6 100

# echo "1---  正在运行: DA3-FiLM 深度引导训练---"
# cd /home/gl/RoboTwin/policy/DP2DP3/features_model
# conda activate depth3
# python tools/depth_guided/train_depth_guided_film_proprio.py  --config configs/depth_guided/train_depth_guided_film_proprio.yaml

# echo "2---  正在运行: DA3-FiLM 深度引导推理---"
# cd /home/gl/RoboTwin
# conda activate RoboTwin
# bash policy/DP2DP3/eval_depth_guided_film.sh lift_pot demo_clean demo_clean 50 0 "0,1" 600 6 100

# echo "3---  正在运行: 2模型直融（CroCo+DINOv3）训练---"
# cd /home/gl/RoboTwin/policy/DP2DP3/features_model
# conda activate depth3
# python tools/direct_fusion/train_direct_fusion_offline_generic_proprio.py \
#     --config configs/direct_fusion/train_direct_fusion_generic_2model_croco_dinov3.yaml

# echo "4---  正在运行: 2模型直融（CroCo+DINOv3）评估---"
# cd /home/gl/RoboTwin
# conda activate RoboTwin
# bash eval_direct_fusion_generic.sh lift_pot demo_clean demo_clean 50 0 "0,1" 600 6 100 croco_dinov3-proprio

# echo "5---  正在运行: 2模型 DA3-FiLM 在线评估---"
# conda activate RoboTwin
# cd /home/gl/RoboTwin/policy/DP2DP3
# bash eval_film_online.sh beat_block_hammer demo_clean demo_clean 50 0 "0" 600 6 100

# echo "6---  正在运行: 3模型 DA3-FiLM 在线训练---"
# python tools/depth_guided_film_online_3model/train_film_online_3model.py   --config configs/depth_guided_film_online_3model/train_film_online_3model.yaml

# echo "7---  正在运行: 3模型 DA3-FiLM 在线评估---"
# conda activate RoboTwin
# cd /home/gl/RoboTwin/policy/DP2DP3
# bash eval_film_online_3model.sh lift_pot demo_clean demo_clean 50 0 "0" 600 6 100

# echo "1---  正在运行: 2模型 DA3-FiLM 在线评估v2---"
# conda activate RoboTwin
# cd /home/gl/RoboTwin
# bash policy/DP2DP3/eval_film_online_v2.sh beat_block_hammer demo_clean demo_clean 50 0 "0" 600 6 100

# echo "2---  正在运行: 2模型 DA3-FiLM 在线评估v1 move can pot---"
# bash policy/DP2DP3/eval_film_online.sh move_can_pot demo_clean demo_clean 50 0 "0" 600 6 100

# echo "3---  Running: DINO-only ablation training on beat_block_hammer---"
# conda activate depth3
# cd /home/gl/RoboTwin/policy/DP2DP3/features_model
# python tools/depth_guided_dino_online/train_dino_online.py \
#     --config configs/depth_guided_dino_online/train_dino_online.yaml

# echo "4---  Running: FiLM temporal-delta training on beat_block_hammer---"
# python tools/depth_guided_film_online_delta/train_film_online_delta.py \
#     --config configs/depth_guided_film_online_delta/train_film_online_delta.yaml


# echo "5---  单dino推理---"
# bash eval_dino_online.sh beat_block_hammer demo_clean demo_clean 50 0 "0" 600 6 100

# echo "6---  FiLM 时序差分评估---"
# bash eval_film_online_delta.sh beat_block_hammer demo_clean demo_clean 50 0 "0" 600 6 100

# CUDA_VISIBLE_DEVICES=1 python tools/depth_guided_film_online/train_film_online.py --config configs/depth_guided_film_online/train_film_online.yaml
# cd /home/gl/RoboTwin/policy/DP2DP3/features_model
# python tools/depth_guided_film_online/train_film_online.py --config configs/depth_guided_film_online/train_film_online.yaml

# cd /home/gl/RoboTwin
# conda activate RoboTwin
# bash policy/DP2DP3/eval_film_online.sh click_bell demo_clean demo_clean 50 0 "0" 600 6 100


# bash policy/DP2DP3/eval_film_online.sh pick_diverse_bottles demo_clean demo_clean 50 0 "0" 600 6 100


# cd /home/gl/RoboTwin/policy/DP2DP3
# eval "$(conda shell.bash hook)"
# conda activate depth3

# CUDA_VISIBLE_DEVICES=0,1 python -u features_model/tools/depth_guided_film_drifting/train_film_drifting.py \
#     --config features_model/configs/depth_guided_film_drifting/train_film_drifting_v18_beat.yaml

# cd /home/gl/RoboTwin
# eval "$(conda shell.bash hook)"
# conda activate RoboTwin

# export DP2DP3_FILM_DRIFTING_CKPT_DIR="/home/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/beat_block_hammer-demo_clean-50-0-v18"
# export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_film_drifting_policy.yml"

# # 评估 ckpt=100, 5轮快速测试
# bash policy/DP2DP3/eval_film_drifting.sh \
#     beat_block_hammer demo_clean demo_clean-50-0-v18 50 0 "0" 100 6 100

# # 评估 ckpt=300, 100轮正式测试
# bash policy/DP2DP3/eval_film_drifting.sh \
#     beat_block_hammer demo_clean demo_clean-50-0-v18 50 0 "0" 300 6 100

# bash policy/DP2DP3/eval_film_drifting.sh \
#     beat_block_hammer demo_clean demo_clean-50-0-v18 50 0 "0" 600 6 100

# cd /home/gl/RoboTwin
# conda activate RoboTwin
# mv /home/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/lift_pot-demo_clean-50-0-v21-ema /home/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/lift_pot-demo_clean-v21-ema-50-0
# export DP2DP3_FILM_DRIFTING_CKPT_DIR="/home/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/lift_pot-demo_clean-v21-ema-50-0"
# export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_film_drifting_policy.yml"
# bash policy/DP2DP3/eval_film_drifting.sh \
#     lift_pot demo_clean demo_clean-v21-ema 50 0 "0" 600 6 100
# bash policy/DP2DP3/eval_film_drifting.sh \
#     lift_pot demo_clean demo_clean-v21-ema 50 0 "0" 300 6 100

# export DP2DP3_FILM_DRIFTING_CKPT_DIR="/home/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/beat_block_hammer-demo_clean-50-0-v18_linear"
# export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_film_drifting_policy.yml"
# bash policy/DP2DP3/eval_film_drifting.sh \
#     beat_block_hammer demo_clean demo_clean-50-0-v18_linear 50 0 "0" 600 6 100
# bash policy/DP2DP3/eval_film_drifting.sh \
#     beat_block_hammer demo_clean demo_clean-50-0-v18_linear 50 0 "0" 300 6 100

# export DP2DP3_FILM_DRIFTING_CKPT_DIR="/home/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/click_bell-demo_clean-50-0-v18"
# export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_film_drifting_policy.yml"
# bash policy/DP2DP3/eval_film_drifting.sh \
#     click_bell demo_clean demo_clean-50-0-v18_linear 50 0 "0" 300 6 100

# bash policy/DP2DP3/eval_film_drifting.sh \
#     click_bell demo_clean demo_clean-50-0-v18_linear 50 0 "0" 600 6 100
# /home/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/lift_pot-demo_clean-v21-ema-50-0/600.ckpt

# cd /home/gl/RoboTwin
# conda activate RoboTwin
# mv /home/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/lift_pot-demo_clean-v22-multipos-alpha1p5 /home/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/lift_pot-demo_clean-v22-multipos-alpha1p5-50-0
# export DP2DP3_FILM_DRIFTING_CKPT_DIR="/home/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/lift_pot-demo_clean-v22-multipos-alpha1p5-50-0"
# export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_film_drifting_policy.yml"
# bash policy/DP2DP3/eval_film_drifting.sh \
#     lift_pot demo_clean demo_clean-v22-multipos-alpha1p5 50 0 "0" 600 6 100
# bash policy/DP2DP3/eval_film_drifting.sh \
#     lift_pot demo_clean demo_clean-v22-multipos-alpha1p5 50 0 "0" 300 6 100
# =====================
# A) 评估 V18-MultiTemp 训练产物 (ckpt=200/300/400/600)
# 训练命令参考:
#   python tools/depth_guided_film_drifting/train_film_drifting_v18_multitemp.py \
#     --config configs/depth_guided_film_drifting/train_film_drifting_v18_multitemp_template.yaml
# # =====================
# conda activate RoboTwin_gl
# cd /data/gl/RoboTwin
# export DP2DP3_DEPLOY_CONFIG="policy/DP2DP3/deploy_film_drifting_policy.yml"

# MULTITEMP_TASK="lift_pot"
# MULTITEMP_TASK_CONFIG="demo_clean"
# MULTITEMP_CKPT_SETTING="demo_clean-v18-multitemp-e600-seed42"
# MULTITEMP_EXPERT_NUM=50
# MULTITEMP_SEED=0
# MULTITEMP_GPU="0"
# MULTITEMP_N_EXEC=6
# MULTITEMP_TEST_NUM=100

# MULTITEMP_CKPT_DIR="/data/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/${MULTITEMP_TASK}-${MULTITEMP_CKPT_SETTING}-${MULTITEMP_EXPERT_NUM}-${MULTITEMP_SEED}"
# if [ -d "$MULTITEMP_CKPT_DIR" ]; then
#   export DP2DP3_FILM_DRIFTING_CKPT_DIR="$MULTITEMP_CKPT_DIR"
# else
#   echo "[WARN] 未找到 multitemp checkpoint 目录: $MULTITEMP_CKPT_DIR"
# fi

# for ckpt in 200 300 400 600; do
#   echo "--- [MultiTemp] eval ckpt=${ckpt} ---"
#   bash policy/DP2DP3/eval_film_drifting.sh \
#     "$MULTITEMP_TASK" "$MULTITEMP_TASK_CONFIG" "$MULTITEMP_CKPT_SETTING" \
#     "$MULTITEMP_EXPERT_NUM" "$MULTITEMP_SEED" "$MULTITEMP_GPU" "$ckpt" \
#     "$MULTITEMP_N_EXEC" "$MULTITEMP_TEST_NUM"
# done


# # =====================
# # B) 评估 film_online 训练产物 (ckpt=600)
# # 训练命令参考:
# #   python tools/depth_guided_film_online/train_film_online.py \
# #     --config configs/depth_guided_film_online/train_film_online.yaml
# # =====================
# FILM_ONLINE_TASK="lift_pot"
# FILM_ONLINE_TASK_CONFIG="demo_clean"
# FILM_ONLINE_CKPT_SETTING="demo_clean"
# FILM_ONLINE_EXPERT_NUM=50
# FILM_ONLINE_SEED=0
# FILM_ONLINE_GPU="0"
# FILM_ONLINE_CKPT=600
# FILM_ONLINE_N_EXEC=6
# FILM_ONLINE_TEST_NUM=100

# echo "--- [FilmOnline] eval ckpt=${FILM_ONLINE_CKPT} ---"
# bash policy/DP2DP3/eval_film_online.sh \
#   "$FILM_ONLINE_TASK" "$FILM_ONLINE_TASK_CONFIG" "$FILM_ONLINE_CKPT_SETTING" \
#   "$FILM_ONLINE_EXPERT_NUM" "$FILM_ONLINE_SEED" "$FILM_ONLINE_GPU" "$FILM_ONLINE_CKPT" \
#   "$FILM_ONLINE_N_EXEC" "$FILM_ONLINE_TEST_NUM"

# echo "[INFO] wait_and_run.sh 已按需求串行执行：MultiTemp(200/300/400/600) + FilmOnline(600)。"



# bash policy/DP2DP3/eval_film_drifting.sh  beat_block_hammer demo_clean demo_clean-v33-prefixbc 50 0 "0" 100 6 100
# bash policy/DP2DP3/eval_film_drifting.sh  beat_block_hammer demo_clean demo_clean-v33-prefixbc 50 0 "0" 200 6 100
# bash policy/DP2DP3/eval_film_drifting.sh  beat_block_hammer demo_clean demo_clean-v33-prefixbc 50 0 "0" 400 6 100
# bash policy/DP2DP3/eval_film_drifting.sh  beat_block_hammer demo_clean demo_clean-v33-prefixbc 50 0 "0" 600 6 100
# bash policy/DP2DP3/eval_film_drifting.sh  click_bell demo_clean demo_clean-v33-prefixbc 50 0 "0" 100 6 100
# bash policy/DP2DP3/eval_film_drifting.sh  move_can_pot demo_clean demo_clean-v18 50 0 "0" 300 6 100
# bash policy/DP2DP3/eval_film_drifting.sh  move_can_pot demo_clean demo_clean-v18 50 0 "0" 600 6 100
# bash policy/DP2DP3/eval_film_drifting.sh stack_bowls_three demo_clean demo_clean-v18 50 0 "0" 300 6 100
# bash policy/DP2DP3/eval_film_drifting.sh place_cans_plasticbox demo_clean demo_clean-v18 50 0 "0" 300 6 100

# cd /data/gl/RoboTwin/policy/FlowPolicy2D
# conda activate RoboTwin_gl
# echo "已切换到 RoboTwin_gl 环境，当前目录: $(pwd)"
# bash eval.sh lift_pot demo_clean demo_clean 50 0 0 600
# cd /data/gl/RoboTwin/policy/DP2DP3/features_model
# conda activate depth3
# python tools/depth_guided_film_drifting/train_film_drifting_v18_mtbcadapt.py --config configs/depth_guided_film_drifting/v18_addons/train_film_drifting_v18_mtbcadapt_template.yaml
# cd /data/gl/RoboTwin
# conda activate RoboTwin_gl
# echo "已切换到 RoboTwin_gl 环境，当前目录: $(pwd)"
# bash eval_film_drifting.sh open_laptop demo_clean demo_clean-v18-multitemp-e600-seed42 50 0 0 600 6 100
# bash eval_film_drifting.sh open_laptop demo_clean demo_clean-v18-multitemp-e600-seed42 50 0 0 300 6 100
# bash eval_film_drifting.sh open_la demo_clean demo_clean-v18-multitemp-e600-seed42 50 0 0 100 6 100

# cd /data/gl/RoboTwin/policy/FlowPolicy2D
# conda activate RoboTwin_gl
# # bash train.sh dump_bin_bigbin demo_clean 50 0 14 0
# bash eval.sh click_bell demo_clean demo_clean 50 0 0 400
# bash eval.sh click_bell demo_clean demo_clean 50 0 0 200
# # bash eval.sh click_bell demo_clean demo_clean 50 0 0 100

# bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 400
# bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 200
# # bash eval.sh lift_pot demo_clean demo_clean 50 0 0 100

# bash eval.sh place_fan demo_clean demo_clean 50 0 0 400
# bash eval.sh place_fan demo_clean demo_clean 50 0 0 200
# # bash eval.sh open_laptop demo_clean demo_clean 50 0 0 100


# cd /data/gl/RoboTwin/policy/DP
# bash eval.sh click_bell demo_clean demo_clean 50 0 0 400
# bash eval.sh click_bell demo_clean demo_clean 50 0 0 200
# # bash eval.sh dump_bin_bigbin demo_clean demo_clean 50 0 0 100

# bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 400
# bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 200
# # bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 100

# bash eval.sh place_fan demo_clean demo_clean 50 0 0 400
# bash eval.sh place_fan demo_clean demo_clean 50 0 0 200
# # bash eval.sh place_fan demo_clean demo_clean 50 0 0 100
# bash eval.sh open_laptop demo_clean demo_clean 50 0 0 100

# cd /data/gl/RoboTwin/policy/MP1_2D
# bash eval.sh click_bell demo_clean demo_clean 50 0 0 400
# bash eval.sh click_bell demo_clean demo_clean 50 0 0 200
# # bash eval.sh click_bell demo_clean demo_clean 50 0 0 100

# bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 400
# bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 200
# # bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 100

# bash eval.sh place_fan demo_clean demo_clean 50 0 0 400
# bash eval.sh place_fan demo_clean demo_clean 50 0 0 200
# bash eval.sh place_fan demo_clean demo_clean 50 0 0 100

# bash eval.sh open_laptop demo_clean demo_clean 50 0 0 400
# bash eval.sh open_laptop demo_clean demo_clean 50 0 0 200
# bash eval.sh open_laptop demo_clean demo_clean 50 0 0 100

# cd /data/gl/RoboTwin/policy/BC_2D
# conda activate RoboTwin_gl
# bash eval.sh click_bell demo_clean demo_clean 50 0 0 400
# bash eval.sh click_bell demo_clean demo_clean 50 0 0 200
# # bash eval.sh click_bell demo_clean demo_clean 50 0 0 100

# bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 400
# bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 200
# # bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 100

# bash eval.sh place_fan demo_clean demo_clean 50 0 0 400
# bash eval.sh place_fan demo_clean demo_clean 50 0 0 200
# # bash eval.sh place_fan demo_clean demo_clean 50 0 0 100

# bash eval.sh open_laptop demo_clean demo_clean 50 0 0 400
# bash eval.sh open_laptop demo_clean demo_clean 50 0 0 200
# # bash eval.sh open_laptop demo_clean demo_clean 50 0 0 100

# bash eval.sh dump_bin_bigbin demo_clean demo_clean 50 0 0 400
# bash eval.sh dump_bin_bigbin demo_clean demo_clean 50 0 0 200
# # bash eval.sh dump_bin_bigbin demo_clean demo_clean 50 0 0 100

# bash eval.sh lift_pot demo_clean demo_clean 50 0 0 400
# bash eval.sh lift_pot demo_clean demo_clean 50 0 0 200
# # bash eval.sh lift_pot demo_clean demo_clean 50 0 0 100

# cd /data/gl/RoboTwin/policy/DP2DP3
# export DP2DP3_DEPLOY_CONFIG=/data/gl/RoboTwin/policy/DP2DP3/deploy_film_drifting_policy_v18_raw_arep.yml
# bash eval.sh stack_bowls_three demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-05-scale32-h6 50 0 "0" 300 6 100
# bash eval.sh stack_bowls_three demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-05-scale32-h6 50 0 "0" 200 6 100


# bash eval_film_drifting.sh stack_bowls_three demo_clean demo_clean-v18-multitemp-e600-seed42 50 0 "0" 600 6 100
# bash eval_film_drifting.sh stack_bowls_three demo_clean demo_clean-v18-multitemp-e600-seed42 50 0 "0" 300 6 100

# conda activate depth3
# python /home/gl/RoboTwin/policy/DP2DP3/features_model/tools/depth_guided_film_drifting/train_film_drifting_v18_mtbcadapt.py --config /home/gl/RoboTwin/policy/DP2DP3/features_model/configs/depth_guided_film_drifting/v18_addons/seed42_mtbc_raw/train_film_drifting_v18_mtbcadapt_raw_open_laptop_seed42.yaml

# bash /home/gl/RoboTwin/policy/DP2DP3/eval_film_drifting.sh open_laptop demo_clean demo_clean-v18-mtbcadapt-raw-e600-seed42 50 0 "1" 100 6 100
# bash /home/gl/RoboTwin/policy/DP2DP3/eval_film_drifting.sh open_laptop demo_clean demo_clean-v18-mtbcadapt-raw-e600-seed42 50 0 "1" 200 6 100
# bash /home/gl/RoboTwin/policy/DP2DP3/eval_film_drifting.sh open_laptop demo_clean demo_clean-v18-mtbcadapt-raw-e600-seed42 50 0 "1" 300 6 100
# bash /home/gl/RoboTwin/policy/DP2DP3/eval_film_drifting.sh open_laptop demo_clean demo_clean-v18-mtbcadapt-raw-e600-seed42 50 0 "1" 400 6 100
# bash /data/gl/RoboTwin/policy/DP2DP3/eval_film_drifting.sh dump_bin_bigbin demo_clean demo_clean-v18-multitemp-paper-e600-seed42 50 0 "0" 100 6 100
# bash /data/gl/RoboTwin/policy/DP2DP3/eval_film_drifting.sh dump_bin_bigbin demo_clean demo_clean-v18-multitemp-paper-e600-seed42 50 0 "0" 300 6 100
# bash /data/gl/RoboTwin/policy/DP2DP3/eval_film_drifting.sh dump_bin_bigbin demo_clean demo_clean-v18-multitemp-paper-e600-seed42 50 0 "0" 600 6 100
# bash eval.sh lift_pot demo_clean demo_clean 50 0 0 600
# bash train.sh click_bell demo_clean 50 0 14 0
# bash eval.sh click_bell demo_clean demo_clean 50 0 0 600
# bash train.sh dump_bin_bigbin demo_clean 50 0 14 0
# bash eval.sh open_laptop demo_clean demo_clean 50 0 0 600
# bash eval.sh dump_bin_bigbin demo_clean demo_clean 50 0 0 600
# cd /data/gl/RoboTwin/DP2DP3/features_model
# conda activate depth3
# python /data/gl/RoboTwin/DP2DP3/features_model/train_film_drifting.py --config configs/depth_guided_film_drifting/v18_addons/seed42_mtbc_raw/train_film_drifting_v18_mtbcadapt_raw_open_laptop_seed42.yaml


cd /data/gl/RoboTwin/policy/DP2DP3
conda activate RoboTwin_gl
# export DP2DP3_DEPLOY_CONFIG=policy/DP2DP3/deploy_film_drifting_policy_v18_raw_arep.yml
# bash eval_film_drifting.sh click_bell demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-scale32-lr3 50 0 "0" 100 6 100
# bash eval_film_drifting.sh click_bell demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-scale32-lr3 50 0 "0" 200 6 100
# bash eval_film_drifting.sh click_bell demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-scale32-lr3 50 0 "0" 300 6 100

# bash eval_film_drifting.sh lift_pot demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 100 6 100
# bash eval_film_drifting.sh lift_pot demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 200 6 100
# bash eval_film_drifting.sh lift_pot demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 300 6 100

# bash eval_film_drifting.sh open_laptop demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 100 6 100
# bash eval_film_drifting.sh open_laptop demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 200 6 100
# bash eval_film_drifting.sh open_laptop demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 300 6 100

# bash eval_film_drifting.sh dump_bin_bigbin demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 100 6 100  
# bash eval_film_drifting.sh dump_bin_bigbin demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 200 6 100
# bash eval_film_drifting.sh dump_bin_bigbin demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 300 6 100

# bash eval_film_drifting.sh stack_bowls_two demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 100 6 100
# bash eval_film_drifting.sh stack_bowls_two demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 200 6 100  
# bash eval_film_drifting.sh stack_bowls_two demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 300 6 100

# bash eval_film_drifting.sh place_fan demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 100 6 100
# bash eval_film_drifting.sh place_fan demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 200 6 100
# bash eval_film_drifting.sh place_fan demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-lr1 50 0 "0" 300 6 100

# cd /data/gl/RoboTwin/policy/DP/data/gl/RoboTwin/policy/DP2DP3/checkpoints_film_drifting/lift_pot-demo_clean-v18-raw-arep-kgate-e300-seed42-lr1-50-0
# conda activate RoboTwin_gl
# bash train.sh dump_bin_bigbin demo_clean 50 0 14 0
# bash eval.sh dump_bin_bigbin demo_clean demo_clean 50 0 0 600

# cd /data/gl/RoboTwin/policy/FlowPolicy2D
# bash train.sh stack_blocks_two demo_clean 50 0 14 0
# bash eval.sh stack_blocks_two demo_clean demo_clean 50 0 0 600

# cd /data/gl/RoboTwin/policy/MP1_2D
# bash train.sh stack_blocks_two demo_clean 50 0 14 0
# bash eval.sh stack_blocks_two demo_clean demo_clean 50 0 0 600
# conda activate RoboTwin_gl

# cd /data/gl/RoboTwin/policy/MP1_2D
# bash train.sh stack_bowls_two demo_clean 50 0 14 0
# bash train.sh place_fan demo_clean 50 0 14 0

# cd /data/gl/RoboTwin/policy/FlowPolicy2D
# bash train.sh stack_bowls_two demo_clean 50 0 14 0
# bash train.sh place_fan demo_clean 50 0 14 0

# cd /data/gl/RoboTwin/policy/MP1_2D
# bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 600
# bash eval.sh place_fan demo_clean demo_clean 50 0 0 600

# cd /data/gl/RoboTwin/policy/FlowPolicy2D
# bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 600
# bash eval.sh place_fan demo_clean demo_clean 50 0 0 600
# cd /data/gl/RoboTwin/policy/BC_2D
# bash train.sh lift_pot demo_clean 50 0 14 0
# bash train.sh open_laptop demo_clean 50 0 14 0
# bash train.sh dump_bin_bigbin demo_clean 50 0 14 0
# bash train.sh stack_bowls_two demo_clean 50 0 14 0
# bash train.sh place_fan demo_clean 50 0 14 0
# bash eval.sh click_bell demo_clean demo_clean 50 0 0 600
# bash eval.sh lift_pot demo_clean demo_clean 50 0 0 600
# bash eval.sh open_laptop demo_clean demo_clean 50 0 0 600
# bash eval.sh dump_bin_bigbin demo_clean demo_clean 50 0 0 600
# bash eval.sh stack_bowls_two demo_clean demo_clean 50 0 0 600
# bash eval.sh place_fan demo_clean demo_clean 50 0 0 600
# conda activate depth3
# python tools/depth_guided_film_drifting/train_film_drifting_v18_raw_arep.py --config configs/depth_guided_film_drifting/train_film_drifting_v18_raw_arep_template.yaml

# cd /data/gl/RoboTwin/policy/DP2DP3
# conda activate RoboTwin_gl
# export DP2DP3_DEPLOY_CONFIG=/data/gl/RoboTwin/policy/DP2DP3/deploy_film_drifting_policy_v18_raw_arep.yml
# bash eval.sh stack_bowls_two demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-05-scale32 50 0 "0" 300 6 100
# bash eval.sh stack_bowls_two demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-05-scale32 50 0 "0" 200 6 100
# bash eval.sh stack_bowls_two demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-05-scale32 50 0 "0" 100 6 100

export WANDB_API_KEY=17f31472e3b5a20dbd1202b0b2efad231ded8e1b
# cd /data/gl/RoboTwin/policy/DP2DP3/features_model
# conda activate depth3
# python tools/depth_guided_film_drifting/train_film_drifting_v18_raw_arep.py --config configs/depth_guided_film_drifting/train_film_drifting_v18_raw_arep_click_bell_seed42.yaml
# conda activate RoboTwin_gl
# cd /data/gl/RoboTwin/policy/DP2DP3
# export DP2DP3_DEPLOY_CONFIG=policy/DP2DP3/deploy_film_drifting_policy_v18_raw_arep.yml
# bash eval_film_drifting.sh click_bell demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-scale50-lr1 50 0 "0" 100 6 100
# bash eval_film_drifting.sh click_bell demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-scale50-lr1 50 0 "0" 200 6 100
# bash eval_film_drifting.sh click_bell demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-scale50-lr1 50 0 "0" 300 6 100

# export WANDB_API_KEY=17f31472e3b5a20dbd1202b0b2efad231ded8e1b
# cd /data/gl/RoboTwin/policy/DP2DP3/features_model
# conda activate depth3
# python tools/depth_guided_film_drifting/train_film_drifting_v18_raw_arep.py --config configs/depth_guided_film_drifting/train_film_drifting_v18_raw_arep_lift_pot_seed42.yaml
# conda activate RoboTwin_gl
# cd /data/gl/RoboTwin/policy/DP2DP3
# export DP2DP3_DEPLOY_CONFIG=policy/DP2DP3/deploy_film_drifting_policy_v18_raw_arep.yml

# bash eval_film_drifting.sh lift_pot demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr1 50 0 "0" 100 6 100
# bash eval_film_drifting.sh lift_pot demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr1 50 0 "0" 200 6 100
# bash eval_film_drifting.sh lift_pot demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr1 50 0 "0" 300 6 100

# export WANDB_API_KEY=17f31472e3b5a20dbd1202b0b2efad231ded8e1b
# cd /data/gl/RoboTwin/policy/DP2DP3/features_model
# conda activate depth3
# python tools/depth_guided_film_drifting/train_film_drifting_v18_raw_arep.py --config configs/depth_guided_film_drifting/train_film_drifting_v18_raw_arep_dump_bin_bigbin_seed42.yaml
conda activate RoboTwin_gl
cd /data/gl/RoboTwin/policy/DP2DP3
export DP2DP3_DEPLOY_CONFIG=policy/DP2DP3/deploy_film_drifting_policy_v18_raw_arep.yml
bash eval_film_drifting.sh dump_bin_bigbin demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr1 50 0 "0" 100 6 100
bash eval_film_drifting.sh dump_bin_bigbin demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr1 50 0 "0" 200 6 100
bash eval_film_drifting.sh dump_bin_bigbin demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr1 50 0 "0" 300 6 100

bash eval_film_drifting.sh place_fan demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr3 50 0 "0" 100 6 100
bash eval_film_drifting.sh place_fan demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr3 50 0 "0" 200 6 100
bash eval_film_drifting.sh place_fan demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr3 50 0 "0" 300 6 100


bash eval_film_drifting.sh place_fan demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr1 50 0 "0" 100 6 100
bash eval_film_drifting.sh place_fan demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr1 50 0 "0" 200 6 100
bash eval_film_drifting.sh place_fan demo_clean demo_clean-v18-raw-arep-kgate-e300-seed0-lr1 50 0 "0" 300 6 100



python tools/depth_guided_film_drifting/train_film_drifting_v18_raw_arep.py --config configs/depth_guided_film_drifting/train_film_drifting_v18_raw_arep_open_laptop_seed42.yaml
python tools/depth_guided_film_drifting/train_film_drifting_v18_raw_arep.py --config configs/depth_guided_film_drifting/train_film_drifting_v18_raw_arep_stack_bowls_two_seed42.yaml
# python tools/depth_guided_film_drifting/train_film_drifting_v18_raw_arep.py --config configs/depth_guided_film_drifting/train_film_drifting_v18_raw_arep_template_seed42.yaml

# conda activate RoboTwin_gl
# cd /data/gl/RoboTwin/policy/DP2DP3
# export DP2DP3_DEPLOY_CONFIG=/data/gl/RoboTwin/policy/DP2DP3/deploy_film_drifting_policy_v18_raw_arep.yml
# bash eval.sh lift_pot demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-si 50 0 "0" 300 6 100
# bash eval.sh lift_pot demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-si 50 0 "0" 200 6 100
# bash eval.sh lift_pot demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-si 50 0 "0" 100 6 100




# conda activate RoboTwin_gl
# cd /data/gl/RoboTwin/policy/DP2DP3
# export DP2DP3_DEPLOY_CONFIG=/data/gl/RoboTwin/policy/DP2DP3/deploy_film_drifting_policy_v18_raw_arep.yml
# bash eval.sh stack_bowls_two demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-scale32-si 50 0 "0" 300 6 100
# bash eval.sh stack_bowls_two demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-scale32-si 50 0 "0" 200 6 100
# bash eval.sh stack_bowls_two demo_clean demo_clean-v18-raw-arep-kgate-e300-seed42-scale32-si 50 0 "0" 100 6 100
# export WANDB_API_KEY=17f31472e3b5a20dbd1202b0b2efad231ded8e1b
# export WANDB_ENTITY=你的_wandb_entity