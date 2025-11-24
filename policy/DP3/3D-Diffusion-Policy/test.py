import torch

# 替换为你的模型文件路径
model_path = "policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/pretrained_model/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt"

# 加载文件（先用cpu避免设备不兼容）
loaded_obj = torch.load(model_path, map_location="cpu")

# 打印加载对象的类型
print(f"加载对象的类型：{type(loaded_obj)}")

# 进一步判断
if isinstance(loaded_obj, torch.nn.Module):
    print("✅ 是「保存了整个模型」（含结构+参数）")
    # 可进一步查看模型结构
    print("模型结构：")
    print(loaded_obj)
elif isinstance(loaded_obj, dict):
    print("❌ 是「仅保存参数」（State Dict或Checkpoint）")
    # 查看字典的键（判断是State Dict还是Checkpoint）
    print("字典的关键键名：", list(loaded_obj.keys())[:10])  # 打印前10个键
    # 若键名含"backbone.weight"、"patch_embed.bias"等，就是模型的state_dict
    # 若键名含"epoch"、"optimizer"、"model_state_dict"等，就是Checkpoint
else:
    print("❌ 既不是完整模型，也不是标准参数字典（可能是自定义格式）")