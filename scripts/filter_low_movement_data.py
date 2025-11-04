import os
import numpy as np
from tqdm import tqdm

input_dir = "../expert_gru_data"
output_dir = "../expert_gru_data_filtered"
os.makedirs(output_dir, exist_ok=True)

displacement_thresh = 0.1
file_id = 0
total = 0
kept = 0
removed = 0

for fname in tqdm(os.listdir(input_dir)):
    if not fname.endswith(".npz"):
        continue

    path = os.path.join(input_dir, fname)
    data = np.load(path)
    obs = data["obs"]         # [N, T, obs_dim]
    action = data["action"]   # [N, T, 2]

    delta_pos = np.linalg.norm(obs[:, -1] - obs[:, 0], axis=1)  # [N]
    keep_mask = delta_pos >= displacement_thresh

    total += len(obs)
    kept += np.sum(keep_mask)
    removed += np.sum(~keep_mask)

    if np.any(keep_mask):
        obs_filtered = obs[keep_mask]
        action_filtered = action[keep_mask]
        save_path = os.path.join(output_dir, f"filtered_{file_id}.npz")
        np.savez(save_path, obs=obs_filtered, action=action_filtered)
        file_id += 1

print("\n✅ 过滤完成！")
print(f"📦 总序列数: {total}")
print(f"✅ 保留序列数: {kept}")
print(f"❌ 丢弃序列数: {removed}")
print(f"📁 保存路径: {output_dir}")

