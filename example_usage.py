#!/usr/bin/env python3
"""
优化集成使用示例
"""

import sys
import os
import torch

# 添加路径
sys.path.append('/home/astwea/MyDogTask/Mydog')
sys.path.append('/home/astwea/IsaacLab/source/isaaclab')

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    from integrated_optimizations import IntegratedOptimizations
    from Mydog.tasks.direct.mydog_marl.mydog_marl_env_cfg import MydogMarlEnvCfg
    
    # 1. 创建配置
    cfg = MydogMarlEnvCfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 创建优化器
    optimizer = IntegratedOptimizations(device, cfg)
    
    # 3. 模拟环境状态
    num_envs = 8
    env_state = {
        'positions': torch.randn(num_envs, 2, device=device),
        'lin_vel': torch.randn(num_envs, 2, device=device),
        'ang_vel': torch.randn(num_envs, 1, device=device),
        'yaw': torch.randn(num_envs, device=device),
        'current_target': torch.randn(num_envs, 2, device=device),
        'next_target': torch.randn(num_envs, 2, device=device),
        'episode_length': torch.randint(0, 100, (num_envs,), device=device),
        'max_episode_length': torch.tensor(100, device=device),
        'current_wp_idx': torch.randint(0, 10, (num_envs,), device=device),
        'total_waypoints': 10,
        'actions': torch.randn(num_envs, 2, device=device),
        'prev_positions': torch.randn(num_envs, 2, device=device),
    }
    
    actions = torch.randn(num_envs, 2, device=device)
    prev_actions = torch.randn(num_envs, 2, device=device)
    
    # 4. 计算优化的观测
    obs = optimizer.get_optimized_observations(env_state, actions, prev_actions)
    print(f"观测形状: {obs.shape}")
    
    # 5. 计算优化的奖励
    total_reward, rewards = optimizer.compute_optimized_rewards(
        env_state, actions, prev_actions
    )
    print(f"总奖励形状: {total_reward.shape}")
    print(f"奖励组件: {list(rewards.keys())}")
    
    return True

def example_training_integration():
    """训练集成示例"""
    print("\n=== 训练集成示例 ===")
    
    from integrated_optimizations import IntegratedOptimizations
    from Mydog.tasks.direct.mydog_marl.mydog_marl_env_cfg import MydogMarlEnvCfg
    
    # 模拟训练循环
    cfg = MydogMarlEnvCfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = IntegratedOptimizations(device, cfg)
    
    # 模拟多个episode
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        
        # 模拟episode数据
        episode_reward = 0
        episode_length = 0
        success = False
        
        for step in range(20):  # 模拟20步
            # 模拟环境状态
            env_state = {
                'positions': torch.randn(4, 2, device=device),
                'lin_vel': torch.randn(4, 2, device=device),
                'ang_vel': torch.randn(4, 1, device=device),
                'yaw': torch.randn(4, device=device),
                'current_target': torch.randn(4, 2, device=device),
                'next_target': torch.randn(4, 2, device=device),
                'episode_length': torch.tensor(step, device=device),
                'max_episode_length': torch.tensor(20, device=device),
                'prev_positions': torch.randn(4, 2, device=device),
            }
            
            actions = torch.randn(4, 2, device=device)
            prev_actions = torch.randn(4, 2, device=device)
            
            # 计算奖励
            total_reward, rewards = optimizer.compute_optimized_rewards(
                env_state, actions, prev_actions
            )
            
            episode_reward += total_reward.mean().item()
            episode_length += 1
        
        # 模拟episode结束
        success = episode_reward > 0  # 简单的成功判断
        
        # 更新性能指标
        optimizer.update_performance_metrics(episode_reward, episode_length, success)
        
        print(f"  - 奖励: {episode_reward:.2f}")
        print(f"  - 长度: {episode_length}")
        print(f"  - 成功: {success}")
    
    # 获取训练建议
    recommendations = optimizer.get_training_recommendations()
    if recommendations:
        print(f"\n训练建议: {recommendations}")
    
    return True

def example_adaptive_parameters():
    """自适应参数示例"""
    print("\n=== 自适应参数示例 ===")
    
    from integrated_optimizations import IntegratedOptimizations
    from Mydog.tasks.direct.mydog_marl.mydog_marl_env_cfg import MydogMarlEnvCfg
    
    cfg = MydogMarlEnvCfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = IntegratedOptimizations(device, cfg)
    
    print("初始参数:")
    print(f"  - 学习率倍数: {optimizer.adaptive_params['learning_rate_multiplier']}")
    print(f"  - 探索倍数: {optimizer.adaptive_params['exploration_multiplier']}")
    
    # 模拟性能数据
    for i in range(10):
        success = i < 5  # 前5个episode成功
        optimizer.update_performance_metrics(10.0 + i, 20, success)
    
    # 自适应参数更新
    optimizer.adaptive_parameter_update()
    
    print("\n更新后参数:")
    print(f"  - 学习率倍数: {optimizer.adaptive_params['learning_rate_multiplier']:.3f}")
    print(f"  - 探索倍数: {optimizer.adaptive_params['exploration_multiplier']:.3f}")
    
    # 获取课程学习难度
    difficulty = optimizer.get_curriculum_difficulty()
    print(f"  - 课程难度: {difficulty:.3f}")
    
    # 获取探索epsilon
    epsilon = optimizer.get_exploration_epsilon(1000)
    print(f"  - 探索epsilon: {epsilon:.3f}")
    
    return True

def example_observation_analysis():
    """观测分析示例"""
    print("\n=== 观测分析示例 ===")
    
    from integrated_optimizations import IntegratedOptimizations
    from Mydog.tasks.direct.mydog_marl.mydog_marl_env_cfg import MydogMarlEnvCfg
    
    cfg = MydogMarlEnvCfg()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = IntegratedOptimizations(device, cfg)
    
    # 获取观测空间信息
    obs_info = optimizer.observation_space.get_observation_info()
    
    print("观测空间信息:")
    print(f"  - 总维度: {obs_info['total_dim']}")
    print(f"  - 组件: {obs_info['dimensions']}")
    
    print("\n组件描述:")
    for key, desc in obs_info['description'].items():
        print(f"  - {key}: {desc}")
    
    return True

def main():
    """主函数"""
    print("🚀 优化集成使用示例")
    print("=" * 50)
    
    examples = [
        example_basic_usage,
        example_training_integration,
        example_adaptive_parameters,
        example_observation_analysis,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"❌ 示例 {example.__name__} 失败: {e}")
    
    print("\n🎉 所有示例运行完成！")

if __name__ == "__main__":
    main()
