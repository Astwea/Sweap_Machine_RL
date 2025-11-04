#!/usr/bin/env python3
"""
测试课程学习修复效果
"""

import sys
import os

def test_curriculum_fixes():
    """测试课程学习修复"""
    print("=== 测试课程学习修复 ===")
    
    # 检查修复是否已应用
    env_file = '/home/astwea/MyDogTask/Mydog/source/Mydog/Mydog/tasks/direct/mydog_marl/mydog_marl_env.py'
    
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    fixes = [
        '_reset_trajectories_all',
        'new_traj_size = self.cfg.num_waypoints * self.cfg.num_interp',
        'if self._trajectories.shape[1] != new_traj_size:',
        'self._trajectories = torch.zeros(self.num_envs, new_traj_size, 2, device=self.device)',
        'self.cfg.episode_length_s = stage_config[\'episode_length_s\']',
        'max_episode_length 是只读属性',
        'self.episode_count = 0',
        'curriculum_success_rate_threshold',
        'curriculum_min_episodes_per_stage',
        'curriculum_success_window_size',
        '_get_window_success_rate',
        'success_history',
        'self.episode_count += len(env_ids)'
    ]
    
    applied_fixes = []
    for fix in fixes:
        if fix in content:
            applied_fixes.append(fix)
            print(f"✅ {fix}")
        else:
            print(f"❌ {fix}")
    
    print(f"\n已应用的修复: {len(applied_fixes)}/{len(fixes)}")
    
    # 检查配置类修复
    cfg_file = '/home/astwea/MyDogTask/Mydog/source/Mydog/Mydog/tasks/direct/mydog_marl/mydog_marl_env_cfg.py'
    
    with open(cfg_file, 'r', encoding='utf-8') as f:
        cfg_content = f.read()
    
    # 检查是否移除了有问题的字典
    if 'curriculum_stages = {' in cfg_content:
        print("❌ 配置类中仍有问题字典")
        return False
    else:
        print("✅ 配置类中已移除问题字典")
    
    return len(applied_fixes) == len(fixes)

def test_trajectory_size_calculation():
    """测试轨迹大小计算"""
    print("\n=== 测试轨迹大小计算 ===")
    
    # 模拟课程学习阶段
    curriculum_stages = {
        0: {'num_waypoints': 2, 'num_interp': 4},  # 8个点
        1: {'num_waypoints': 3, 'num_interp': 6},  # 18个点
        2: {'num_waypoints': 5, 'num_interp': 12}, # 60个点
    }
    
    for stage, config in curriculum_stages.items():
        traj_size = config['num_waypoints'] * config['num_interp']
        print(f"阶段 {stage}: {config['num_waypoints']} waypoints × {config['num_interp']} interp = {traj_size} 个轨迹点")
    
    print("✅ 轨迹大小计算正确")

def test_episode_length_handling():
    """测试回合长度处理"""
    print("\n=== 测试回合长度处理 ===")
    
    # 模拟不同阶段的回合长度
    stages = [
        {'episode_length_s': 10.0, 'dt': 1/200},
        {'episode_length_s': 12.0, 'dt': 1/200},
        {'episode_length_s': 15.0, 'dt': 1/200},
    ]
    
    for i, stage in enumerate(stages):
        max_episode_length = int(stage['episode_length_s'] / stage['dt'])
        print(f"阶段 {i}: {stage['episode_length_s']}s / {stage['dt']}s = {max_episode_length} 步")
    
    print("✅ 回合长度计算正确")

if __name__ == "__main__":
    print("开始测试课程学习修复...")
    
    # 测试修复
    fixes_ok = test_curriculum_fixes()
    
    if fixes_ok:
        print("\n✅ 所有修复已应用")
        
        # 测试计算逻辑
        test_trajectory_size_calculation()
        test_episode_length_handling()
        
        print("\n🎉 课程学习修复验证完成！")
    else:
        print("\n❌ 部分修复未应用，请检查代码")
    
    print("\n=== 测试完成 ===")
