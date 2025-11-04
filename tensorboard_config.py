#!/usr/bin/env python3
"""
TensorBoard 监控配置和工具
用于 MyDog 强化学习训练过程的全面监控
"""

import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Any, Optional

class TensorBoardMonitor:
    """增强版 TensorBoard 监控器"""
    
    def __init__(self, log_dir: str, experiment_name: str = "mydog_rl"):
        """
        初始化 TensorBoard 监控器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
        
        # 监控指标分类
        self.metrics_categories = {
            "Reward": [
                "progress_reward", "lateral_penalty", "direction_reward", 
                "goal_bias", "action_rate_penalty", "action_mag_penalty", 
                "imitation_reward", "Total"
            ],
            "Environment": [
                "Distance_to_Target", "Lateral_Error", "Heading_Error",
                "Robot_Speed", "Action_Magnitude", "Action_Rate"
            ],
            "Performance": [
                "Episode_Length", "Total_Distance", "Average_Speed",
                "Max_Lateral_Error", "Step_Time", "FPS"
            ],
            "Episode": [
                "Episode_Reward", "Episode_Length", "Success_Rate",
                "Total_Distance", "Final_Distance", "Max_Lateral_Error"
            ],
            "Training": [
                "Best_Reward", "Average_Reward", "Total_Episodes",
                "Elapsed_Time", "Steps_Per_Second"
            ]
        }
        
        # 统计信息
        self.stats = {
            "episode_count": 0,
            "total_steps": 0,
            "best_reward": float('-inf'),
            "start_time": time.time()
        }
    
    def log_reward_metrics(self, rewards: Dict[str, torch.Tensor], step: int):
        """记录奖励指标"""
        for key, value in rewards.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(f"Reward/{key}", value.mean().item(), step)
            else:
                self.writer.add_scalar(f"Reward/{key}", value, step)
    
    def log_environment_metrics(self, metrics: Dict[str, Any], step: int):
        """记录环境指标"""
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(f"Environment/{key}", value.mean().item(), step)
            else:
                self.writer.add_scalar(f"Environment/{key}", value, step)
    
    def log_performance_metrics(self, metrics: Dict[str, Any], step: int):
        """记录性能指标"""
        for key, value in metrics.items():
            self.writer.add_scalar(f"Performance/{key}", value, step)
    
    def log_episode_summary(self, episode_data: Dict[str, Any], episode_num: int):
        """记录回合总结"""
        for key, value in episode_data.items():
            self.writer.add_scalar(f"Episode/{key}", value, episode_num)
        
        # 更新统计信息
        self.stats["episode_count"] = episode_num
        if "Episode_Reward" in episode_data:
            self.stats["best_reward"] = max(self.stats["best_reward"], episode_data["Episode_Reward"])
    
    def log_training_progress(self, step: int):
        """记录训练进度"""
        elapsed_time = time.time() - self.stats["start_time"]
        steps_per_second = step / elapsed_time if elapsed_time > 0 else 0
        
        self.writer.add_scalar("Training/Elapsed_Time", elapsed_time, step)
        self.writer.add_scalar("Training/Steps_Per_Second", steps_per_second, step)
        self.writer.add_scalar("Training/Best_Reward", self.stats["best_reward"], step)
        self.writer.add_scalar("Training/Total_Episodes", self.stats["episode_count"], step)
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """记录超参数"""
        self.writer.add_hparams(
            hparam_dict=config,
            metric_dict={"best_reward": self.stats["best_reward"]}
        )
    
    def log_model_architecture(self, model):
        """记录模型架构"""
        # 这里可以添加模型架构的可视化
        pass
    
    def log_distributions(self, data: Dict[str, torch.Tensor], step: int):
        """记录数据分布"""
        for key, values in data.items():
            if isinstance(values, torch.Tensor) and values.numel() > 0:
                self.writer.add_histogram(f"Distributions/{key}", values, step)
    
    def close(self):
        """关闭监控器"""
        self.writer.close()

def create_tensorboard_dashboard_config():
    """创建 TensorBoard 仪表板配置"""
    config = {
        "version": 1,
        "disable_theme": False,
        "timezone": "Asia/Shanghai",
        "scalars": {
            "layout": {
                "height": 300,
                "margin": 5
            },
            "smoothing": 0.6,
            "xAxis": "step",
            "yAxis": "value"
        },
        "histograms": {
            "layout": {
                "height": 300,
                "margin": 5
            },
            "smoothing": 0.6
        }
    }
    return config

def print_monitoring_guide():
    """打印监控指南"""
    print("""
🎯 MyDog 强化学习 TensorBoard 监控指南
========================================

📊 主要监控指标分类:

1. 奖励指标 (Reward/*)
   - progress_reward: 路径推进奖励
   - lateral_penalty: 侧向误差惩罚
   - direction_reward: 方向奖励
   - goal_bias: 目标偏差奖励
   - action_rate_penalty: 动作变化率惩罚
   - action_mag_penalty: 动作幅度惩罚
   - imitation_reward: 模仿学习奖励
   - Total: 总奖励

2. 环境指标 (Environment/*)
   - Distance_to_Target: 到目标距离
   - Lateral_Error: 侧向误差
   - Heading_Error: 航向误差
   - Robot_Speed: 机器人速度
   - Action_Magnitude: 动作幅度
   - Action_Rate: 动作变化率

3. 性能指标 (Performance/*)
   - Episode_Length: 回合长度
   - Total_Distance: 总行驶距离
   - Average_Speed: 平均速度
   - Max_Lateral_Error: 最大侧向误差
   - Step_Time: 每步执行时间
   - FPS: 每秒帧数

4. 回合统计 (Episode/*)
   - Episode_Reward: 回合总奖励
   - Episode_Length: 回合长度
   - Success_Rate: 成功率
   - Total_Distance: 总距离
   - Final_Distance: 最终距离
   - Max_Lateral_Error: 最大侧向误差

5. 训练统计 (Training/*)
   - Best_Reward: 最佳奖励
   - Average_Reward: 平均奖励
   - Total_Episodes: 总回合数
   - Elapsed_Time: 已用时间
   - Steps_Per_Second: 每秒步数

🔧 使用建议:
- 关注 Reward/Total 的上升趋势
- 监控 Environment/Lateral_Error 的下降
- 观察 Performance/FPS 保持稳定
- 检查 Episode/Success_Rate 的提升
- 分析 Training/Best_Reward 的增长

📈 优化建议:
- 如果奖励不增长，检查学习率设置
- 如果侧向误差大，调整奖励权重
- 如果FPS低，考虑减少环境数量
- 如果成功率低，增加训练时间
""")

if __name__ == "__main__":
    print_monitoring_guide()
