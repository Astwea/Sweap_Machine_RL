import os
import numpy as np
from tqdm import tqdm

data_dir = "../expert_gru_data_filtered"

velocity_thresh = 0.01
displacement_thresh = 0.1
angular_thresh = 0.05

total_seqs = 0
stuck_by_action = 0
stuck_by_movement = 0
stuck_both = 0

for fname in tqdm(os.listdir(data_dir)):
    if not fname.endswith(".npz"):
        continue

    data = np.load(os.path.join(data_dir, fname))
    obs = data["obs"]         # [N, T, obs_dim]
    action = data["action"]   # [N, T, 2]
    v = np.abs(action[:, :, 0])
    w = np.abs(action[:, :, 1])

    # 判断整条序列动作是否都非常小
    stuck_action_mask = np.all((v < velocity_thresh) & (w < angular_thresh), axis=1)

    # 判断观测位移是否过小
    delta_pos = np.linalg.norm(obs[:, -1] - obs[:, 0], axis=1)
    stuck_move_mask = delta_pos < displacement_thresh

    stuck_by_action += np.sum(stuck_action_mask)
    stuck_by_movement += np.sum(stuck_move_mask)
    stuck_both += np.sum(stuck_action_mask & stuck_move_mask)
    total_seqs += len(obs)

print("\n===== 📊 卡住动作数据统计（动作 + 位移） =====")
print(f"📦 总序列数: {total_seqs}")
print(f"🛑 全程低速动作序列数: {stuck_by_action} ({stuck_by_action / total_seqs * 100:.2f}%)")
print(f"🐢 位移极小序列数: {stuck_by_movement} ({stuck_by_movement / total_seqs * 100:.2f}%)")
print(f"❌ 两者都满足（完全卡住）序列数: {stuck_both} ({stuck_both / total_seqs * 100:.2f}%)")

