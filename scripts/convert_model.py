# convert_model.py
import torch

old_path = 'logs/rl_games/diff_drive_direct/2025-05-22_22-03-51/nn/diff_drive_direct.pth'
new_path = 'logs/rl_games/diff_drive_direct/2025-05-22_22-03-51/nn/convert.pth'

M = 6

# 假设原输入维度是 D，新的变为 D+M
old_model = torch.load(old_path)
print(old_model.keys())         # 看 top-level 是不是包含 'model'
print(old_model['model'].keys())
old_weights = old_model['model']

# 假设你用了 actor_critic 的 MLP，权重名可能是：
# 'actor.fc1.weight', 'critic.fc1.weight' 等

# 举例扩展 actor 第一个全连接层
old_fc1 = old_weights['a2c_network.actor_mlp.0.weight']  # [hidden_dim, old_dim]
hidden_dim, old_dim = old_fc1.shape
new_dim = old_dim + M  # 新的输入维度

# 创建新参数，扩展为 [hidden_dim, new_dim]
new_fc1 = torch.zeros((hidden_dim, new_dim))
new_fc1[:, :old_dim] = old_fc1  # 前半部分复制原权重
old_weights['actor.fc1.weight'] = new_fc1

# 你也可能需要处理 bias，或 critic 层等

# 保存新模型
torch.save({'model': old_weights}, new_path)

