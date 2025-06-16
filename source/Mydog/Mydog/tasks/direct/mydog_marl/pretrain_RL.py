import numpy as np
import pickle
from .mydog_marl_env import MydogMarlEnv  # 根据你的环境导入

# 1. 初始化环境
env = MydogMarlEnv()  # env_cfg按你的配置
dataset = []

num_episodes = 100
for ep in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        # === 2. 用你的MPC做专家控制 ===
        action = mpc_controller(obs)  # TODO: 你要实现
        # === 3. 执行动作，获取下一步 ===
        next_obs, reward, done, info = env.step(action)
        # === 4. 存下本步的数据 ===
        dataset.append({
            'obs': obs,  # 通常存ndarray
            'action': action,
            'next_obs': next_obs,
            'reward': reward,
            'done': done
        })
        obs = next_obs
    print(f"Episode {ep+1}/{num_episodes} collected.")

# 5. 保存为pkl或npz
with open('mpc_expert_data.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print("专家轨迹采集完成，保存在 mpc_expert_data.pkl")
