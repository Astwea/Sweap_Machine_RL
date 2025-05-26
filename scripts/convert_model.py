import torch

import torch

def expand_model_and_optimizer(pth_path, save_path, input_expand=6):
    ckpt = torch.load(pth_path, map_location="cpu")
    model_weights = ckpt["model"]

    key = "a2c_network.actor_mlp.0.weight"
    old_w = model_weights[key]
    old_shape = old_w.shape
    new_shape = (old_shape[0], old_shape[1] + input_expand)

    # 扩展模型权重
    new_w = torch.zeros(new_shape)
    new_w[:, :old_shape[1]] = old_w
    model_weights[key] = new_w
    ckpt["model"] = model_weights

    # 同时扩展 optimizer 参数
    optim_state = ckpt.get("optimizer", {}).get("state", {})
    for param_id, state in optim_state.items():
        for stat_key in ["exp_avg", "exp_avg_sq"]:
            if stat_key in state and state[stat_key].shape == old_shape:
                expanded = torch.zeros(new_shape)
                expanded[:, :old_shape[1]] = state[stat_key]
                state[stat_key] = expanded
                print(f"[✓] 扩展 optimizer.{param_id}.{stat_key} 到 {new_shape}")

    # 保存新模型
    torch.save(ckpt, save_path)
    print(f"\n✅ 成功扩展输入维度并保存至：{save_path}")

# 使用示例
if __name__ == "__main__":
    expand_model_and_optimizer(
        pth_path="logs/rl_games/diff_drive_direct/2025-05-22_22-03-51/nn/diff_drive_direct.pth",
        save_path="logs/rl_games/diff_drive_direct/2025-05-22_22-03-51/nn/converted21.pth",
        input_expand=6
    )
