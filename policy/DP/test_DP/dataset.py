# 运行独立脚本查看数据集结构
import zarr
z = zarr.open("data/beat_block_hammer-demo_trans-50_multi_cam.zarr", 'r')
print("数据集包含的键：", list(z["data"].keys()))