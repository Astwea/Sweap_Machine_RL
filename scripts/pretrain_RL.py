import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# === 你的网络结构（不变）===
class D2RLNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(obs_dim + hidden_dim, hidden_dim)
        self.act = nn.ELU()

    def forward(self, x):
        x1 = self.act(self.fc1(x))
        x2_in = torch.cat([x, x1], dim=-1)
        x2 = self.act(self.fc2(x2_in))
        return x2

class BCPolicyGRU(nn.Module):
    def __init__(self, obs_dim=25, act_dim=2):
        super().__init__()
        self.actor_mlp = D2RLNet(obs_dim)
        self.rnn = nn.GRU(128, 128, batch_first=True)
        self.mu = nn.Linear(128, act_dim)

    def forward(self, obs_seq):
        B, T, _ = obs_seq.shape
        obs_seq = obs_seq.reshape(B * T, -1)
        mlp_out = self.actor_mlp(obs_seq)  # [B*T, 128]
        mlp_out = mlp_out.view(B, T, -1)   # [B, T, 128]
        rnn_out, _ = self.rnn(mlp_out)
        act_out = self.mu(rnn_out)         # [B, T, act_dim]
        return act_out

# === Dataset ===
class GRUSeqDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        for file in os.listdir(data_dir):
            if file.endswith(".npz"):
                data = np.load(os.path.join(data_dir, file))
                obs_arr = data["obs"]      # (2048, 4, 25)
                action_arr = data["action"]  # (2048, 4, 2)
                for obs_seq, act_seq in zip(obs_arr, action_arr):
                    self.samples.append((obs_seq, act_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, act = self.samples[idx]
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(act, dtype=torch.float32)

# === Main training ===
def train_bc(data_dir="../expert_gru_data", obs_dim=25, act_dim=2, seq_len=4, batch_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 加载并拆分数据集
    full_dataset = GRUSeqDataset(data_dir)
    total_len = len(full_dataset)
    val_len = int(0.1 * total_len)
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 2. 模型 & 优化器
    model = BCPolicyGRU(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(100):
        model.train()
        total_train_loss = 0
        for obs_seq, act_seq in train_loader:
            obs_seq, act_seq = obs_seq.to(device), act_seq.to(device)
            pred = model(obs_seq)
            loss = loss_fn(pred, act_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # === 验证集评估 ===
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for obs_seq, act_seq in val_loader:
                obs_seq, act_seq = obs_seq.to(device), act_seq.to(device)
                pred = model(obs_seq)
                loss = loss_fn(pred, act_seq)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"[Epoch {epoch}] Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        # === 保存最优权重 ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "bc_gru_best.pth")
            print(f"✅ [Epoch {epoch}] Saved new best model bc_gru_best.pth")

    print("✅ Training complete.")

    # === 加载最佳模型，评估测试集 ===
    model.load_state_dict(torch.load("bc_gru_best.pth"))
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for obs_seq, act_seq in val_loader:
            obs_seq, act_seq = obs_seq.to(device), act_seq.to(device)
            pred = model(obs_seq)
            loss = loss_fn(pred, act_seq)
            total_val_loss += loss.item()

    print(f"🎯 Final Validation Loss (Best Model): {total_val_loss / len(val_loader):.6f}")

if __name__ == "__main__":
    train_bc("../expert_gru_data_filtered")
