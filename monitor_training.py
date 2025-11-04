#!/usr/bin/env python3
"""
训练监控脚本
实时监控 MyDog 强化学习训练进度
"""

import os
import time
import glob
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.metrics = {}
        self.last_update = 0
        
    def find_latest_log(self):
        """查找最新的日志目录"""
        log_dirs = glob.glob(os.path.join(self.log_dir, "*"))
        if not log_dirs:
            return None
        return max(log_dirs, key=os.path.getctime)
    
    def load_tensorboard_data(self, log_path: str):
        """加载 TensorBoard 数据"""
        try:
            ea = EventAccumulator(log_path)
            ea.Reload()
            
            # 获取所有标量数据
            scalar_tags = ea.Tags()['scalars']
            data = {}
            
            for tag in scalar_tags:
                scalar_events = ea.Scalars(tag)
                steps = [event.step for event in scalar_events]
                values = [event.value for event in scalar_events]
                data[tag] = {'steps': steps, 'values': values}
            
            return data
        except Exception as e:
            print(f"加载 TensorBoard 数据失败: {e}")
            return {}
    
    def print_training_summary(self, data: dict):
        """打印训练摘要"""
        print("\n" + "="*60)
        print("🎯 MyDog 强化学习训练监控")
        print("="*60)
        
        # 获取最新数据
        latest_data = {}
        for tag, values in data.items():
            if values['values']:
                latest_data[tag] = values['values'][-1]
        
        # 显示关键指标
        if 'Reward/Total' in latest_data:
            print(f"📊 当前总奖励: {latest_data['Reward/Total']:.4f}")
        
        if 'Training/Best_Reward' in latest_data:
            print(f"🏆 最佳奖励: {latest_data['Training/Best_Reward']:.4f}")
        
        if 'Training/Total_Episodes' in latest_data:
            print(f"🎮 总回合数: {int(latest_data['Training/Total_Episodes'])}")
        
        if 'Performance/FPS' in latest_data:
            print(f"⚡ 当前FPS: {latest_data['Performance/FPS']:.1f}")
        
        if 'Environment/Lateral_Error' in latest_data:
            print(f"🎯 侧向误差: {latest_data['Environment/Lateral_Error']:.4f}")
        
        if 'Episode/Success_Rate' in latest_data:
            print(f"✅ 成功率: {latest_data['Episode/Success_Rate']:.2%}")
        
        print("="*60)
    
    def plot_training_curves(self, data: dict, save_path: str = None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('MyDog 强化学习训练监控', fontsize=16)
        
        # 奖励曲线
        if 'Reward/Total' in data:
            axes[0, 0].plot(data['Reward/Total']['steps'], data['Reward/Total']['values'])
            axes[0, 0].set_title('总奖励')
            axes[0, 0].set_xlabel('步数')
            axes[0, 0].set_ylabel('奖励')
            axes[0, 0].grid(True)
        
        # 侧向误差
        if 'Environment/Lateral_Error' in data:
            axes[0, 1].plot(data['Environment/Lateral_Error']['steps'], data['Environment/Lateral_Error']['values'])
            axes[0, 1].set_title('侧向误差')
            axes[0, 1].set_xlabel('步数')
            axes[0, 1].set_ylabel('误差')
            axes[0, 1].grid(True)
        
        # 成功率
        if 'Episode/Success_Rate' in data:
            axes[0, 2].plot(data['Episode/Success_Rate']['steps'], data['Episode/Success_Rate']['values'])
            axes[0, 2].set_title('成功率')
            axes[0, 2].set_xlabel('回合数')
            axes[0, 2].set_ylabel('成功率')
            axes[0, 2].grid(True)
        
        # FPS
        if 'Performance/FPS' in data:
            axes[1, 0].plot(data['Performance/FPS']['steps'], data['Performance/FPS']['values'])
            axes[1, 0].set_title('FPS')
            axes[1, 0].set_xlabel('步数')
            axes[1, 0].set_ylabel('FPS')
            axes[1, 0].grid(True)
        
        # 回合奖励
        if 'Episode/Episode_Reward' in data:
            axes[1, 1].plot(data['Episode/Episode_Reward']['steps'], data['Episode/Episode_Reward']['values'])
            axes[1, 1].set_title('回合奖励')
            axes[1, 1].set_xlabel('回合数')
            axes[1, 1].set_ylabel('奖励')
            axes[1, 1].grid(True)
        
        # 训练进度
        if 'Training/Total_Episodes' in data:
            axes[1, 2].plot(data['Training/Total_Episodes']['steps'], data['Training/Total_Episodes']['values'])
            axes[1, 2].set_title('训练进度')
            axes[1, 2].set_xlabel('步数')
            axes[1, 2].set_ylabel('回合数')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 训练曲线已保存到: {save_path}")
        else:
            plt.show()
    
    def monitor_loop(self, interval: int = 30, plot: bool = False):
        """监控循环"""
        print("🚀 开始监控训练过程...")
        print(f"📁 监控目录: {self.log_dir}")
        print(f"⏱️  更新间隔: {interval}秒")
        print("按 Ctrl+C 停止监控\n")
        
        try:
            while True:
                latest_log = self.find_latest_log()
                if latest_log:
                    data = self.load_tensorboard_data(latest_log)
                    if data:
                        self.print_training_summary(data)
                        
                        if plot:
                            plot_path = os.path.join(self.log_dir, "training_curves.png")
                            self.plot_training_curves(data, plot_path)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 监控已停止")

def main():
    parser = argparse.ArgumentParser(description="MyDog 强化学习训练监控")
    parser.add_argument("--log_dir", type=str, default="/home/astwea/MyDogTask/Mydog/runs/logs", help="日志目录")
    parser.add_argument("--interval", type=int, default=30, help="更新间隔(秒)")
    parser.add_argument("--plot", action="store_true", help="生成训练曲线图")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_dir)
    monitor.monitor_loop(args.interval, args.plot)

if __name__ == "__main__":
    main()
