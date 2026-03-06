#!/bin/bash
# DP2DP3 数据处理脚本
#
# 用法:
#   bash process_data.sh <task_name> <task_config> <expert_data_num> [cameras...]
#
# 示例:
#   bash process_data.sh beat_block_hammer demo_clean 50
#   bash process_data.sh move_can_pot demo_clean 50
#   bash process_data.sh lift_pot demo_clean 50
#
#   # 提取多个摄像头:
#   bash process_data.sh beat_block_hammer demo_clean 50 head_camera front_camera

task_name=${1:?请指定任务名}
task_config=${2:?请指定配置名}
expert_data_num=${3:?请指定episode数}
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -gt 0 ]; then
    python "${SCRIPT_DIR}/process_data.py" "$task_name" "$task_config" "$expert_data_num" --cameras "$@"
else
    python "${SCRIPT_DIR}/process_data.py" "$task_name" "$task_config" "$expert_data_num"
fi
