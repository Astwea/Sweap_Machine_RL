"""
优化的奖励函数设计
"""

import torch
import math

class OptimizedRewardFunction:
    """优化的奖励函数类"""
    
    def __init__(self, device, config):
        self.device = device
        self.config = config
        
        # 奖励权重（可调参数）
        self.weights = {
            'progress': 15.0,      # 路径推进奖励权重
            'lateral': 25.0,       # 侧向误差惩罚权重
            'heading': 20.0,       # 朝向奖励权重
            'goal': 30.0,          # 目标到达奖励权重
            'efficiency': 5.0,     # 效率奖励权重
            'smoothness': 8.0,     # 平滑性奖励权重
            'time_penalty': 0.1,   # 时间惩罚权重
            'imitation': 10.0,     # 模仿学习权重
            'curvature': 5.0,      # 曲率平滑权重
        }
        
        # 历史奖励记录（用于自适应调整）
        self.reward_history = {key: [] for key in self.weights.keys()}
        
    def compute_rewards(self, env_state, actions, prev_actions, teacher_actions=None):
        """计算优化后的奖励"""
        rewards = {}
        
        # 1. 路径推进奖励（改进版）
        progress_reward = self._compute_progress_reward(env_state)
        rewards['progress'] = progress_reward
        
        # 2. 侧向误差惩罚（改进版）
        lateral_penalty = self._compute_lateral_penalty(env_state)
        rewards['lateral'] = -lateral_penalty
        
        # 3. 朝向奖励（改进版）
        heading_reward = self._compute_heading_reward(env_state)
        rewards['heading'] = heading_reward
        
        # 4. 目标到达奖励（改进版）
        goal_reward = self._compute_goal_reward(env_state)
        rewards['goal'] = goal_reward
        
        # 5. 效率奖励（新增）
        efficiency_reward = self._compute_efficiency_reward(env_state, actions)
        rewards['efficiency'] = efficiency_reward
        
        # 6. 平滑性奖励（改进版）
        smoothness_reward = self._compute_smoothness_reward(actions, prev_actions)
        rewards['smoothness'] = smoothness_reward
        
        # 7. 时间惩罚（新增）
        time_penalty = self._compute_time_penalty(env_state)
        rewards['time_penalty'] = -time_penalty
        
        # 8. 模仿学习奖励（修复版）
        if teacher_actions is not None:
            imitation_reward = self._compute_imitation_reward(actions, teacher_actions)
            rewards['imitation'] = imitation_reward
        else:
            rewards['imitation'] = torch.zeros_like(progress_reward)
            
        # 9. 曲率平滑奖励（新增）
        curvature_reward = self._compute_curvature_reward(env_state)
        rewards['curvature'] = curvature_reward
        
        # 计算总奖励
        total_reward = sum(
            self.weights[key] * reward * self.config.step_dt 
            for key, reward in rewards.items()
        )
        
        return total_reward, rewards
    
    def _compute_progress_reward(self, env_state):
        """改进的路径推进奖励"""
        pos = env_state['positions']
        vel = env_state['velocities']
        current_target = env_state['current_target']
        next_target = env_state['next_target']
        yaw = env_state['yaw']
        
        # 计算路径方向
        path_direction = next_target - current_target
        path_direction_norm = torch.norm(path_direction, dim=1, keepdim=True) + 1e-6
        path_direction_unit = path_direction / path_direction_norm
        
        # 计算机器人朝向
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        
        # 计算切向速度
        v_forward = torch.sum(vel * path_direction_unit, dim=1)
        
        # 计算朝向对齐度
        alignment = torch.sum(heading * path_direction_unit, dim=1)
        
        # 使用更平滑的奖励函数
        progress_reward = v_forward * alignment * torch.exp(-torch.abs(v_forward) * 0.5)
        
        return progress_reward
    
    def _compute_lateral_penalty(self, env_state):
        """改进的侧向误差惩罚"""
        pos = env_state['positions']
        current_target = env_state['current_target']
        next_target = env_state['next_target']
        
        # 计算到路径线段的距离
        ab = next_target - current_target
        ap = pos - current_target
        
        # 投影到路径上
        t = torch.clamp(torch.sum(ap * ab, dim=1) / (torch.sum(ab * ab, dim=1) + 1e-6), 0.0, 1.0)
        proj = current_target + t.unsqueeze(1) * ab
        
        # 侧向距离
        lateral_error = torch.norm(pos - proj, dim=1)
        
        # 使用指数惩罚，避免过度惩罚
        lateral_penalty = lateral_error * torch.exp(-lateral_error * 2.0)
        
        return lateral_penalty
    
    def _compute_heading_reward(self, env_state):
        """改进的朝向奖励"""
        pos = env_state['positions']
        current_target = env_state['current_target']
        next_target = env_state['next_target']
        yaw = env_state['yaw']
        
        # 计算目标朝向
        target_direction = next_target - current_target
        target_heading = torch.atan2(target_direction[:, 1], target_direction[:, 0])
        
        # 计算朝向误差
        heading_error = yaw - target_heading
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
        
        # 使用余弦函数，更平滑
        heading_reward = torch.cos(heading_error)
        
        return heading_reward
    
    def _compute_goal_reward(self, env_state):
        """改进的目标到达奖励"""
        pos = env_state['positions']
        current_target = env_state['current_target']
        dist_to_target = torch.norm(pos - current_target, dim=1)
        
        # 分层奖励设计
        goal_reward = torch.zeros_like(dist_to_target)
        
        # 接近奖励
        close_mask = dist_to_target < 0.5
        goal_reward[close_mask] = 0.3 * (0.5 - dist_to_target[close_mask]) / 0.5
        
        # 到达奖励
        reached_mask = dist_to_target < 0.2
        goal_reward[reached_mask] = 1.0
        
        return goal_reward
    
    def _compute_efficiency_reward(self, env_state, actions):
        """效率奖励：鼓励快速完成任务"""
        pos = env_state['positions']
        current_target = env_state['current_target']
        dist_to_target = torch.norm(pos - current_target, dim=1)
        
        # 基于距离的效率奖励
        efficiency = torch.exp(-dist_to_target * 2.0)
        
        return efficiency
    
    def _compute_smoothness_reward(self, actions, prev_actions):
        """改进的平滑性奖励"""
        # 动作变化率
        action_diff = actions - prev_actions
        action_rate = torch.norm(action_diff, dim=1)
        
        # 动作幅度
        action_magnitude = torch.norm(actions, dim=1)
        
        # 组合平滑性奖励
        smoothness = torch.exp(-action_rate * 2.0) * torch.exp(-action_magnitude * 0.5)
        
        return smoothness
    
    def _compute_time_penalty(self, env_state):
        """时间惩罚：鼓励快速完成"""
        episode_length = env_state['episode_length']
        max_episode_length = env_state['max_episode_length']
        
        # 线性时间惩罚
        time_penalty = episode_length.float() / max_episode_length
        
        return time_penalty
    
    def _compute_imitation_reward(self, actions, teacher_actions):
        """修复的模仿学习奖励"""
        # 计算动作差异
        action_diff = actions - teacher_actions
        imitation_loss = torch.sum(action_diff ** 2, dim=1)
        
        # 转换为奖励（越小越好）
        imitation_reward = torch.exp(-imitation_loss * 5.0)
        
        return imitation_reward
    
    def _compute_curvature_reward(self, env_state):
        """曲率平滑奖励"""
        if 'prev_positions' not in env_state:
            return torch.zeros(env_state['positions'].shape[0], device=self.device)
            
        pos = env_state['positions']
        prev_pos = env_state['prev_positions']
        
        # 计算曲率（简化版）
        velocity = pos - prev_pos
        velocity_norm = torch.norm(velocity, dim=1) + 1e-6
        
        # 曲率平滑奖励
        curvature_reward = torch.exp(-velocity_norm * 0.1)
        
        return curvature_reward
    
    def update_weights(self, performance_metrics):
        """根据性能指标自适应调整权重"""
        # 这里可以实现自适应权重调整逻辑
        pass
