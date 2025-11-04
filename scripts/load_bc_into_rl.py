import torch

from pretrain_RL import BCPolicyGRU  # 你的 BC 模型定义

def load_and_merge_bc_weights(bc_path, rl_ckpt_path, save_path):
    # 加载 BC 权重
    bc_model = BCPolicyGRU(obs_dim=25, act_dim=2)
    bc_model.load_state_dict(torch.load(bc_path, map_location='cpu'))
    bc_state = bc_model.state_dict()

    # 加载 RL 权重
    rl_ckpt = torch.load(rl_ckpt_path, map_location='cpu')
    rl_state = rl_ckpt['model']

    # 明确映射
    key_mapping = {
        'actor_mlp.fc1.weight': 'a2c_network.actor_mlp.linears.0.weight',
        'actor_mlp.fc1.bias': 'a2c_network.actor_mlp.linears.0.bias',
        'actor_mlp.fc2.weight': 'a2c_network.actor_mlp.linears.1.weight',
        'actor_mlp.fc2.bias': 'a2c_network.actor_mlp.linears.1.bias',
        'rnn.weight_ih_l0': 'a2c_network.rnn.rnn.weight_ih_l0',
        'rnn.weight_hh_l0': 'a2c_network.rnn.rnn.weight_hh_l0',
        'rnn.bias_ih_l0': 'a2c_network.rnn.rnn.bias_ih_l0',
        'rnn.bias_hh_l0': 'a2c_network.rnn.rnn.bias_hh_l0',
        'mu.weight': 'a2c_network.mu.weight',
        'mu.bias': 'a2c_network.mu.bias',
    }

    updated = []
    for bc_k, rl_k in key_mapping.items():
        if rl_k in rl_state and bc_k in bc_state:
            if rl_state[rl_k].shape != bc_state[bc_k].shape:
                print(f"[跳过] shape 不一致: {rl_k}")
                continue
            rl_state[rl_k] = bc_state[bc_k]
            updated.append(rl_k)

    print(f"✅ 成功更新 {len(updated)} 个参数：")
    for k in updated:
        print(f" - {k}")

    torch.save(rl_ckpt, save_path)
    print(f"✅ 保存新模型到：{save_path}")

if __name__ == "__main__":
    load_and_merge_bc_weights(
        rl_ckpt_path="/home/astwea/MyDogTask/Mydog/logs/rl_games/diff_drive_direct/2025-06-27_04-16-43/nn/diff_drive_direct1.pth",
        bc_path="/home/astwea/MyDogTask/Mydog/scripts/bc_gru_best.pth",
        save_path="/home/astwea/MyDogTask/Mydog/logs/rl_games/diff_drive_direct/2025-06-27_04-16-43/nn/diff_drive_direct.pth",
    )

