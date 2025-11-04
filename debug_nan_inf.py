#!/usr/bin/env python3
"""
数值稳定性调试工具
用于检测和修复训练过程中的NaN和Inf问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class NaNInfDebugger:
    def __init__(self, device='cuda'):
        self.device = device
        self.nan_count = 0
        self.inf_count = 0
        self.debug_log = []
        
    def check_tensor(self, tensor, name="unknown", step=0, replace_with_zero=True):
        """检查张量中的NaN和Inf值"""
        if not isinstance(tensor, torch.Tensor):
            return tensor
            
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        if has_nan or has_inf:
            print(f"[Step {step}] {name}: NaN={has_nan}, Inf={has_inf}")
            self.debug_log.append({
                'step': step,
                'name': name,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'shape': tensor.shape,
                'min': tensor.min().item() if not has_nan and not has_inf else float('nan'),
                'max': tensor.max().item() if not has_nan and not has_inf else float('nan'),
                'mean': tensor.mean().item() if not has_nan and not has_inf else float('nan'),
                'std': tensor.std().item() if not has_nan and not has_inf else float('nan')
            })
            
            if has_nan:
                self.nan_count += 1
            if has_inf:
                self.inf_count += 1
                
            if replace_with_zero:
                tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), 
                                   torch.zeros_like(tensor), tensor)
        
        return tensor
    
    def check_quaternion(self, quat, name="quaternion", step=0):
        """检查四元数的数值稳定性"""
        if not isinstance(quat, torch.Tensor):
            return quat
            
        # 检查NaN和Inf
        quat = self.check_tensor(quat, name, step)
        
        # 检查四元数归一化
        norm = torch.sqrt(torch.sum(quat**2, dim=-1))
        if torch.any(norm < 1e-6):
            print(f"[Step {step}] {name}: 四元数接近零向量")
            # 重新归一化
            norm = torch.clamp(norm, min=1e-8)
            quat = quat / norm.unsqueeze(-1)
        
        return quat
    
    def check_angle(self, angle, name="angle", step=0):
        """检查角度的数值稳定性"""
        if not isinstance(angle, torch.Tensor):
            return angle
            
        # 检查NaN和Inf
        angle = self.check_tensor(angle, name, step)
        
        # 将角度限制在合理范围内
        angle = torch.atan2(torch.sin(angle), torch.cos(angle))
        
        return angle
    
    def check_reward_components(self, rewards, step=0):
        """检查奖励组件的数值稳定性"""
        if not isinstance(rewards, dict):
            return rewards
            
        for key, reward in rewards.items():
            rewards[key] = self.check_tensor(reward, f"reward_{key}", step)
            
        return rewards
    
    def generate_report(self):
        """生成调试报告"""
        print(f"\n=== 数值稳定性调试报告 ===")
        print(f"NaN 检测次数: {self.nan_count}")
        print(f"Inf 检测次数: {self.inf_count}")
        print(f"总检测次数: {len(self.debug_log)}")
        
        if self.debug_log:
            print(f"\n=== 详细日志 ===")
            for entry in self.debug_log[-10:]:  # 显示最后10条
                print(f"Step {entry['step']}: {entry['name']} "
                      f"(NaN={entry['has_nan']}, Inf={entry['has_inf']}) "
                      f"shape={entry['shape']}")
        
        return self.debug_log
    
    def plot_debug_timeline(self, save_path=None):
        """绘制调试时间线"""
        if not self.debug_log:
            print("没有调试数据可绘制")
            return
            
        steps = [entry['step'] for entry in self.debug_log]
        nan_steps = [entry['step'] for entry in self.debug_log if entry['has_nan']]
        inf_steps = [entry['step'] for entry in self.debug_log if entry['has_inf']]
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(steps, bins=50, alpha=0.7, label='总检测')
        plt.hist(nan_steps, bins=50, alpha=0.7, label='NaN检测')
        plt.hist(inf_steps, bins=50, alpha=0.7, label='Inf检测')
        plt.xlabel('训练步数')
        plt.ylabel('检测次数')
        plt.title('数值问题检测时间线')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        problem_types = ['NaN', 'Inf', 'Both']
        problem_counts = [
            sum(1 for entry in self.debug_log if entry['has_nan'] and not entry['has_inf']),
            sum(1 for entry in self.debug_log if entry['has_inf'] and not entry['has_nan']),
            sum(1 for entry in self.debug_log if entry['has_nan'] and entry['has_inf'])
        ]
        plt.pie(problem_counts, labels=problem_types, autopct='%1.1f%%')
        plt.title('数值问题类型分布')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"调试图表已保存到: {save_path}")
        
        plt.show()

def test_nan_inf_detection():
    """测试NaN和Inf检测功能"""
    debugger = NaNInfDebugger()
    
    # 创建测试数据
    test_data = {
        'normal': torch.randn(10, 5),
        'with_nan': torch.tensor([1.0, float('nan'), 3.0, 4.0, 5.0]),
        'with_inf': torch.tensor([1.0, float('inf'), 3.0, 4.0, 5.0]),
        'with_both': torch.tensor([1.0, float('nan'), float('inf'), 4.0, 5.0])
    }
    
    print("=== 测试数值稳定性检测 ===")
    for name, data in test_data.items():
        print(f"\n测试 {name}:")
        print(f"原始数据: {data}")
        cleaned = debugger.check_tensor(data, name, 0)
        print(f"清理后: {cleaned}")
    
    # 生成报告
    debugger.generate_report()

if __name__ == "__main__":
    test_nan_inf_detection()
