"""
集成优化方案：将奖励、观测和模型优化整合到环境中
"""

import torch
import numpy as np
from typing import Dict, Any, Optional

class IntegratedOptimizations:
    """集成优化类"""
    
    def __init__(self, device, config):
        self.device = device
        self.config = config
        
        # 初始化各个优化模块
        from optimized_rewards import OptimizedRewardFunction
        from optimized_observations import OptimizedObservationSpace
        
        self.reward_function = OptimizedRewardFunction(device, config)
        self.observation_space = OptimizedObservationSpace(device, config)
        
        # 性能监控
        self.performance_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'efficiency': []
        }
        
        # 自适应参数
        self.adaptive_params = {
            'reward_weights': self.reward_function.weights.copy(),
            'learning_rate_multiplier': 1.0,
            'exploration_multiplier': 1.0
        }
    
    def get_optimized_observations(self, env_state: Dict[str, Any], 
                                 actions: torch.Tensor, 
                                 prev_actions: torch.Tensor) -> torch.Tensor:
        """获取优化的观测"""
        return self.observation_space.get_observations(env_state, actions, prev_actions)
    
    def compute_optimized_rewards(self, env_state: Dict[str, Any], 
                                actions: torch.Tensor, 
                                prev_actions: torch.Tensor,
                                teacher_actions: Optional[torch.Tensor] = None) -> tuple:
        """计算优化的奖励"""
        return self.reward_function.compute_rewards(env_state, actions, prev_actions, teacher_actions)
    
    def update_performance_metrics(self, episode_reward: float, 
                                 episode_length: int, 
                                 success: bool):
        """更新性能指标"""
        self.performance_metrics['episode_rewards'].append(episode_reward)
        self.performance_metrics['episode_lengths'].append(episode_length)
        self.performance_metrics['success_rate'].append(1.0 if success else 0.0)
        
        # 计算效率（奖励/时间）
        efficiency = episode_reward / max(episode_length, 1)
        self.performance_metrics['efficiency'].append(efficiency)
        
        # 保持最近1000个episode的记录
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > 1000:
                self.performance_metrics[key] = self.performance_metrics[key][-1000:]
    
    def adaptive_parameter_update(self):
        """自适应参数更新"""
        if len(self.performance_metrics['episode_rewards']) < 100:
            return
        
        # 计算最近性能
        recent_rewards = self.performance_metrics['episode_rewards'][-100:]
        recent_success_rate = np.mean(self.performance_metrics['success_rate'][-100:])
        recent_efficiency = np.mean(self.performance_metrics['efficiency'][-100:])
        
        # 根据性能调整奖励权重
        if recent_success_rate < 0.3:  # 成功率低
            self.adaptive_params['reward_weights']['goal'] *= 1.1
            self.adaptive_params['reward_weights']['progress'] *= 1.05
        elif recent_success_rate > 0.8:  # 成功率高
            self.adaptive_params['reward_weights']['smoothness'] *= 1.1
            self.adaptive_params['reward_weights']['efficiency'] *= 1.05
        
        # 根据效率调整学习率
        if recent_efficiency < np.mean(self.performance_metrics['efficiency'][-200:-100]):
            self.adaptive_params['learning_rate_multiplier'] *= 0.95
        else:
            self.adaptive_params['learning_rate_multiplier'] *= 1.02
        
        # 限制参数范围
        self.adaptive_params['learning_rate_multiplier'] = np.clip(
            self.adaptive_params['learning_rate_multiplier'], 0.1, 2.0
        )
    
    def get_curriculum_difficulty(self) -> float:
        """获取课程学习难度"""
        if len(self.performance_metrics['success_rate']) < 50:
            return 0.2  # 初始难度
        
        recent_success_rate = np.mean(self.performance_metrics['success_rate'][-50:])
        
        # 根据成功率调整难度
        if recent_success_rate > 0.8:
            return min(1.0, 0.2 + (recent_success_rate - 0.8) * 4)
        elif recent_success_rate < 0.3:
            return max(0.1, 0.2 - (0.3 - recent_success_rate) * 0.5)
        else:
            return 0.2 + (recent_success_rate - 0.3) * 1.2
    
    def get_exploration_epsilon(self, step: int) -> float:
        """获取探索epsilon值"""
        epsilon_start = 0.9
        epsilon_end = 0.05
        epsilon_decay = 0.995
        
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * (epsilon_decay ** step)
        return epsilon * self.adaptive_params['exploration_multiplier']
    
    def should_early_stop(self) -> bool:
        """判断是否应该早停"""
        if len(self.performance_metrics['episode_rewards']) < 100:
            return False
        
        recent_rewards = self.performance_metrics['episode_rewards'][-50:]
        if len(recent_rewards) < 50:
            return False
        
        # 如果最近50个episode的奖励没有显著提升，考虑早停
        if len(self.performance_metrics['episode_rewards']) >= 100:
            prev_rewards = self.performance_metrics['episode_rewards'][-100:-50]
            if np.mean(recent_rewards) <= np.mean(prev_rewards) * 1.01:  # 提升小于1%
                return True
        
        return False
    
    def get_training_recommendations(self) -> Dict[str, Any]:
        """获取训练建议"""
        recommendations = {}
        
        if len(self.performance_metrics['episode_rewards']) < 50:
            return {"status": "insufficient_data"}
        
        recent_success_rate = np.mean(self.performance_metrics['success_rate'][-50:])
        recent_efficiency = np.mean(self.performance_metrics['efficiency'][-50:])
        recent_rewards = self.performance_metrics['episode_rewards'][-50:]
        
        # 成功率建议
        if recent_success_rate < 0.3:
            recommendations['success_rate'] = {
                'issue': 'low_success_rate',
                'suggestion': 'Increase goal reward weight and reduce task difficulty',
                'action': 'Adjust reward weights and curriculum difficulty'
            }
        elif recent_success_rate > 0.9:
            recommendations['success_rate'] = {
                'issue': 'high_success_rate',
                'suggestion': 'Increase task difficulty or add more complex scenarios',
                'action': 'Implement curriculum learning with harder tasks'
            }
        
        # 效率建议
        if recent_efficiency < np.mean(self.performance_metrics['efficiency'][-100:-50]):
            recommendations['efficiency'] = {
                'issue': 'decreasing_efficiency',
                'suggestion': 'Agent may be overfitting, consider regularization',
                'action': 'Increase dropout, weight decay, or reduce learning rate'
            }
        
        # 奖励稳定性建议
        reward_std = np.std(recent_rewards)
        if reward_std > np.mean(recent_rewards) * 0.5:
            recommendations['stability'] = {
                'issue': 'high_reward_variance',
                'suggestion': 'High variance in rewards, consider reward shaping',
                'action': 'Smooth reward function or add more consistent rewards'
            }
        
        return recommendations
    
    def log_performance_summary(self, writer, global_step: int):
        """记录性能摘要"""
        if len(self.performance_metrics['episode_rewards']) < 10:
            return
        
        # 计算统计信息
        recent_rewards = self.performance_metrics['episode_rewards'][-100:]
        recent_lengths = self.performance_metrics['episode_lengths'][-100:]
        recent_success_rate = np.mean(self.performance_metrics['success_rate'][-100:])
        recent_efficiency = np.mean(self.performance_metrics['efficiency'][-100:])
        
        # 记录到tensorboard
        writer.add_scalar("Performance/Mean_Reward", np.mean(recent_rewards), global_step)
        writer.add_scalar("Performance/Reward_Std", np.std(recent_rewards), global_step)
        writer.add_scalar("Performance/Mean_Length", np.mean(recent_lengths), global_step)
        writer.add_scalar("Performance/Success_Rate", recent_success_rate, global_step)
        writer.add_scalar("Performance/Efficiency", recent_efficiency, global_step)
        
        # 记录自适应参数
        for key, value in self.adaptive_params.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    writer.add_scalar(f"Adaptive/{key}_{sub_key}", sub_value, global_step)
            else:
                writer.add_scalar(f"Adaptive/{key}", value, global_step)
    
    def reset(self):
        """重置优化器状态"""
        self.performance_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'efficiency': []
        }
        
        self.adaptive_params = {
            'reward_weights': self.reward_function.weights.copy(),
            'learning_rate_multiplier': 1.0,
            'exploration_multiplier': 1.0
        }
        
        # 重置历史缓冲区
        self.observation_space.history_buffer = {
            'positions': [],
            'actions': [],
            'velocities': [],
            'max_history': 5
        }
