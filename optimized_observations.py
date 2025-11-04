"""
优化的观测空间设计
"""

import torch
import numpy as np

class OptimizedObservationSpace:
    """优化的观测空间类"""
    
    def __init__(self, device, config):
        self.device = device
        self.config = config
        
        # 观测维度定义（实际计算得出的维度）
        self.dimensions = {
            'velocity': 3,           # 线速度归一化(2) + 角速度(1)
            'actions': 6,            # 当前动作(2) + 前一步动作(2) + 动作变化率(2)
            'relative_error': 3,     # 相对位置误差(2) + 距离误差(1)
            'trajectory': 6,         # 当前目标点(2) + 下一个目标点(2) + 轨迹方向(2)
            'pose': 4,               # 位置(2) + cos_yaw(1) + sin_yaw(1)
            'history': 6,            # 位置历史(2) + 动作历史(2) + 速度历史(2)
            'environment': 4,        # 碰撞(1) + 地面接触(1) + 复杂度(1) + 时间进度(1)
            'progress': 2,           # 路径点进度(1) + 距离进度(1)
        }
        
        self.total_dim = sum(self.dimensions.values())
        
        # 历史缓冲区
        self.history_buffer = {
            'positions': [],
            'actions': [],
            'velocities': [],
            'max_history': 5  # 保留5步历史
        }
    
    def get_observations(self, env_state, actions, prev_actions):
        """获取优化的观测"""
        observations = []
        
        # 1. 速度信息
        vel_obs = self._get_velocity_observations(env_state)
        observations.append(vel_obs)
        
        # 2. 动作信息（包含历史）
        action_obs = self._get_action_observations(actions, prev_actions)
        observations.append(action_obs)
        
        # 3. 相对误差
        error_obs = self._get_error_observations(env_state)
        observations.append(error_obs)
        
        # 4. 轨迹信息（优化版）
        traj_obs = self._get_trajectory_observations(env_state)
        observations.append(traj_obs)
        
        # 5. 姿态信息
        pose_obs = self._get_pose_observations(env_state)
        observations.append(pose_obs)
        
        # 6. 历史信息
        history_obs = self._get_history_observations(env_state)
        observations.append(history_obs)
        
        # 7. 环境状态
        env_obs = self._get_environment_observations(env_state)
        observations.append(env_obs)
        
        # 8. 进度信息
        progress_obs = self._get_progress_observations(env_state)
        observations.append(progress_obs)
        
        # 合并所有观测
        obs = torch.cat(observations, dim=-1)
        
        # 更新历史缓冲区
        self._update_history_buffer(env_state, actions)
        
        # 数据验证和清理
        obs = self._validate_observations(obs)
        
        return obs
    
    def _get_velocity_observations(self, env_state):
        """速度观测"""
        lin_vel = env_state['lin_vel']  # [N, 2]
        ang_vel = env_state['ang_vel']  # [N, 1]
        
        # 速度归一化
        lin_vel_norm = torch.norm(lin_vel, dim=1, keepdim=True) + 1e-6
        lin_vel_normalized = lin_vel / lin_vel_norm
        
        return torch.cat([lin_vel_normalized, ang_vel], dim=-1)
    
    def _get_action_observations(self, actions, prev_actions):
        """动作观测（包含历史）"""
        # 当前动作和前一步动作
        action_obs = torch.cat([actions, prev_actions], dim=-1)
        
        # 添加动作变化率
        action_diff = actions - prev_actions
        action_obs = torch.cat([action_obs, action_diff], dim=-1)
        
        return action_obs
    
    def _get_error_observations(self, env_state):
        """误差观测"""
        pos = env_state['positions']
        current_target = env_state['current_target']
        
        # 相对位置误差
        relative_error = current_target - pos
        
        # 距离误差
        distance_error = torch.norm(relative_error, dim=1, keepdim=True)
        
        return torch.cat([relative_error, distance_error], dim=-1)
    
    def _get_trajectory_observations(self, env_state):
        """轨迹观测（优化版）"""
        current_target = env_state['current_target']
        next_target = env_state['next_target']
        future_targets = env_state.get('future_targets', [])
        
        # 当前目标点
        traj_obs = current_target
        
        # 下一个目标点
        traj_obs = torch.cat([traj_obs, next_target], dim=-1)
        
        # 轨迹方向
        if len(future_targets) > 0:
            future_target = future_targets[0]
            direction = future_target - current_target
            direction_norm = torch.norm(direction, dim=1, keepdim=True) + 1e-6
            direction_unit = direction / direction_norm
            traj_obs = torch.cat([traj_obs, direction_unit], dim=-1)
        else:
            # 如果没有未来目标，使用零向量
            traj_obs = torch.cat([traj_obs, torch.zeros_like(current_target)], dim=-1)
        
        return traj_obs
    
    def _get_pose_observations(self, env_state):
        """姿态观测"""
        pos = env_state['positions']
        yaw = env_state['yaw']
        
        # 位置（相对坐标）
        pose_obs = pos
        
        # 朝向（sin/cos表示）
        cos_yaw = torch.cos(yaw).unsqueeze(1)
        sin_yaw = torch.sin(yaw).unsqueeze(1)
        
        return torch.cat([pose_obs, cos_yaw, sin_yaw], dim=-1)
    
    def _get_history_observations(self, env_state):
        """历史信息观测"""
        if len(self.history_buffer['positions']) < 2:
            # 如果历史不足，用零填充
            return torch.zeros(env_state['positions'].shape[0], 6, device=self.device)
        
        # 位置历史
        pos_history = self.history_buffer['positions'][-2:]  # 最近2步
        pos_diff = pos_history[1] - pos_history[0]
        
        # 动作历史
        if len(self.history_buffer['actions']) >= 2:
            action_history = self.history_buffer['actions'][-2:]
            action_diff = action_history[1] - action_history[0]
        else:
            action_diff = torch.zeros_like(env_state['actions'])
        
        # 速度历史
        if len(self.history_buffer['velocities']) >= 2:
            vel_history = self.history_buffer['velocities'][-2:]
            vel_diff = vel_history[1] - vel_history[0]
        else:
            vel_diff = torch.zeros_like(env_state['lin_vel'])
        
        return torch.cat([pos_diff, action_diff, vel_diff], dim=-1)
    
    def _get_environment_observations(self, env_state):
        """环境状态观测"""
        # 碰撞检测（如果有的话）
        collision = env_state.get('collision', torch.zeros(env_state['positions'].shape[0], 1, device=self.device))
        
        # 地面接触（如果有的话）
        ground_contact = env_state.get('ground_contact', torch.ones(env_state['positions'].shape[0], 1, device=self.device))
        
        # 环境复杂度（基于轨迹曲率）
        trajectory_complexity = self._compute_trajectory_complexity(env_state)
        
        # 时间信息
        episode_progress = env_state['episode_length'].float() / env_state['max_episode_length']
        
        return torch.cat([collision, ground_contact, trajectory_complexity, episode_progress.unsqueeze(1)], dim=-1)
    
    def _get_progress_observations(self, env_state):
        """进度信息观测"""
        current_wp_idx = env_state['current_wp_idx']
        total_wp = env_state['total_waypoints']
        
        # 当前路径点进度
        wp_progress = current_wp_idx.float() / total_wp
        
        # 到当前目标点的距离进度
        pos = env_state['positions']
        current_target = env_state['current_target']
        dist_to_target = torch.norm(pos - current_target, dim=1)
        max_dist = 5.0  # 假设最大距离
        dist_progress = torch.clamp(1.0 - dist_to_target / max_dist, 0.0, 1.0)
        
        return torch.stack([wp_progress, dist_progress], dim=1)
    
    def _compute_trajectory_complexity(self, env_state):
        """计算轨迹复杂度"""
        if 'trajectory_points' not in env_state:
            return torch.zeros(env_state['positions'].shape[0], 1, device=self.device)
        
        # 简化的复杂度计算：基于轨迹点的变化
        traj_points = env_state['trajectory_points']
        if len(traj_points) < 3:
            return torch.zeros(env_state['positions'].shape[0], 1, device=self.device)
        
        # 计算曲率变化
        complexity = torch.zeros(env_state['positions'].shape[0], 1, device=self.device)
        
        return complexity
    
    def _update_history_buffer(self, env_state, actions):
        """更新历史缓冲区"""
        # 添加当前位置
        self.history_buffer['positions'].append(env_state['positions'].clone())
        
        # 添加当前动作
        self.history_buffer['actions'].append(actions.clone())
        
        # 添加当前速度
        self.history_buffer['velocities'].append(env_state['lin_vel'].clone())
        
        # 保持缓冲区大小
        for key in self.history_buffer:
            if key != 'max_history':
                if len(self.history_buffer[key]) > self.history_buffer['max_history']:
                    self.history_buffer[key].pop(0)
    
    def _validate_observations(self, obs):
        """验证和清理观测数据"""
        # 检查NaN
        if torch.isnan(obs).any():
            print("Warning: NaN found in observations, replacing with zeros")
            obs = torch.where(torch.isnan(obs), torch.zeros_like(obs), obs)
        
        # 检查Inf
        if torch.isinf(obs).any():
            print("Warning: Inf found in observations, replacing with zeros")
            obs = torch.where(torch.isinf(obs), torch.zeros_like(obs), obs)
        
        # 检查数值范围
        obs = torch.clamp(obs, -10.0, 10.0)
        
        return obs
    
    def get_observation_info(self):
        """获取观测空间信息"""
        return {
            'total_dim': self.total_dim,
            'dimensions': self.dimensions,
            'description': {
                'velocity': '线速度和角速度',
                'angular_velocity': '角速度',
                'actions': '当前动作、前一步动作和动作变化率',
                'relative_error': '相对位置误差和距离误差',
                'trajectory': '当前目标点、下一个目标点和轨迹方向',
                'pose': '位置和朝向信息',
                'history': '位置、动作和速度的历史变化',
                'environment': '碰撞、接触、复杂度和时间信息',
                'progress': '路径点进度和距离进度'
            }
        }
