#!/bin/bash
# 快速检查点云生成进度。是用来监控 process_rgb_pcd.py（点云生成脚本）的运行进度的。
# 如果您还在运行点云生成任务，它是有用的

echo "==== 进程状态 ===="
ps aux | grep "process_rgb_pcd.py" | grep -v grep || echo "没有运行中的进程"

echo ""
echo "==== 已完成的episode数量 ===="
for cam in front_camera head_camera left_camera right_camera; do
    dir="rgbpc_dataset/PC/beat_block_hammer-demo_randomized-50_${cam}"
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" 2>/dev/null | grep "episode_" | wc -l)
        echo "$cam: $count episodes"
    fi
done

echo ""
echo "==== 最新日志 ===="
if [ -f "process_all.log" ]; then
    tail -5 process_all.log
else
    echo "日志文件不存在"
fi

echo ""
echo "==== 存储使用情况 ===="
if [ -d "rgbpc_dataset" ]; then
    du -sh rgbpc_dataset/PC/* 2>/dev/null | head -4
else
    echo "数据目录不存在"
fi
