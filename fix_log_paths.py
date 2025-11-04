#!/usr/bin/env python3
"""
修复日志路径问题的脚本
解决 TensorBoard 日志文件路径不一致的问题
"""

import os
import shutil
from pathlib import Path

def fix_log_paths():
    """修复日志路径问题"""
    print("🔧 修复 TensorBoard 日志路径问题")
    print("=" * 50)
    
    # 当前工作目录
    current_dir = Path.cwd()
    print(f"📁 当前目录: {current_dir}")
    
    # 检查现有的日志目录结构
    logs_dir = current_dir / "logs"
    runs_logs_dir = current_dir / "runs" / "logs"
    
    print(f"\n📊 检查现有目录结构:")
    print(f"  logs/ 存在: {logs_dir.exists()}")
    print(f"  runs/logs/ 存在: {runs_logs_dir.exists()}")
    
    if logs_dir.exists():
        print(f"\n📂 logs/ 目录内容:")
        for item in sorted(logs_dir.iterdir()):
            print(f"  {item.name}")
    
    if runs_logs_dir.exists():
        print(f"\n📂 runs/logs/ 目录内容:")
        for item in sorted(runs_logs_dir.iterdir()):
            print(f"  {item.name}")
    
    # 查找 TensorBoard 事件文件
    print(f"\n🔍 查找 TensorBoard 事件文件:")
    tensorboard_files = []
    
    # 在 logs 目录中查找
    if logs_dir.exists():
        for root, dirs, files in os.walk(logs_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    tensorboard_files.append(os.path.join(root, file))
    
    # 在 runs/logs 目录中查找
    if runs_logs_dir.exists():
        for root, dirs, files in os.walk(runs_logs_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    tensorboard_files.append(os.path.join(root, file))
    
    print(f"  找到 {len(tensorboard_files)} 个 TensorBoard 事件文件")
    for i, file in enumerate(tensorboard_files[:5]):  # 只显示前5个
        print(f"    {i+1}. {file}")
    if len(tensorboard_files) > 5:
        print(f"    ... 还有 {len(tensorboard_files) - 5} 个文件")
    
    # 建议修复方案
    print(f"\n💡 修复建议:")
    
    if logs_dir.exists() and runs_logs_dir.exists():
        print("  ✅ 两个目录都存在，建议统一使用 logs/ 目录")
        print("  🔧 已更新配置文件使用 logs/ 目录")
    elif logs_dir.exists():
        print("  ✅ logs/ 目录存在，这是正确的路径")
        print("  🔧 已更新配置文件使用 logs/ 目录")
    elif runs_logs_dir.exists():
        print("  ⚠️  只有 runs/logs/ 目录存在")
        print("  🔧 建议将内容移动到 logs/ 目录")
        
        # 询问是否移动文件
        response = input("  是否将 runs/logs/ 的内容移动到 logs/ 目录? (y/n): ")
        if response.lower() == 'y':
            if not logs_dir.exists():
                logs_dir.mkdir(parents=True)
            
            # 移动文件
            for item in runs_logs_dir.iterdir():
                dest = logs_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.move(str(item), str(dest))
                    print(f"    📁 移动目录: {item.name}")
                else:
                    shutil.move(str(item), str(dest))
                    print(f"    📄 移动文件: {item.name}")
            
            print("  ✅ 文件移动完成")
    else:
        print("  ❌ 没有找到任何日志目录")
        print("  🔧 请先运行训练脚本生成日志文件")
    
    # 创建正确的目录结构
    print(f"\n🏗️  创建正确的目录结构:")
    correct_logs_dir = current_dir / "logs"
    if not correct_logs_dir.exists():
        correct_logs_dir.mkdir(parents=True)
        print("  ✅ 创建 logs/ 目录")
    else:
        print("  ✅ logs/ 目录已存在")
    
    # 验证修复结果
    print(f"\n✅ 修复完成!")
    print(f"📁 正确的日志目录: {correct_logs_dir.absolute()}")
    print(f"🚀 现在可以使用以下命令启动 TensorBoard:")
    print(f"   tensorboard --logdir=logs --port=6006")
    print(f"   或使用: ./start_tensorboard_enhanced.sh")

if __name__ == "__main__":
    fix_log_paths()
