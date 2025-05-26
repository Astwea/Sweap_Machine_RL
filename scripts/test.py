import torch

pth_path = "logs/rl_games/diff_drive_direct/2025-05-22_22-03-51/nn/diff_drive_direct.pth"
checkpoint = torch.load(pth_path, map_location='cpu')

# 兼容性提取模型权重
state_dict = checkpoint.get("model", checkpoint)

print("参数层名称和形状:")
for k, v in state_dict.items():
    print(f"{k:50s} -> {tuple(v.shape)}")
