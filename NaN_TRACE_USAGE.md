# NaN/Inf 溯源系统使用说明

## 功能概述

系统现在可以自动检测并详细记录所有 NaN/Inf 的出现，包括：
- 变量名称
- 出现位置（文件、行号、函数名、代码行）
- 详细的统计信息（NaN数量、比例、有效值范围等）
- 首次出现位置
- 完整的事件日志

## 自动溯源

系统会在检测到 NaN/Inf 时自动打印详细信息，包括：

```
================================================================================
⚠️  NaN/Inf 检测到! 变量: reward
================================================================================
步骤: 1234 | 总计出现次数: 1
NaN数量: 5/48 (10.42%)
Inf数量: 0/48 (0.00%)
形状: torch.Size([48]) | 类型: torch.float32 | 设备: cuda:0
有效值范围: [-0.123456, 0.789012]
有效值均值: 0.234567 ± 0.123456

调用位置:
  文件: /path/to/mydog_marl_env.py
  行号: 567
  函数: _get_rewards
  代码: reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

🔍 这是变量 'reward' 第一次出现 NaN/Inf
================================================================================
```

## 手动查询溯源信息

### 1. 打印所有变量的溯源摘要

```python
# 在训练过程中或训练后
env.print_nan_trace_summary()
```

### 2. 查询特定变量的溯源

```python
# 只查看 'reward' 变量的溯源
env.print_nan_trace_summary(variable_name='reward')
```

### 3. 获取溯源数据（编程方式）

```python
# 获取完整的溯源摘要字典
summary = env.get_nan_trace_summary()

# 获取特定变量的溯源
reward_summary = env.get_nan_trace_summary(variable_name='reward')

# 包含的信息：
# - total_events: 总事件数
# - variables_with_nan: 所有出现NaN的变量列表
# - first_occurrences: 每个变量首次出现的详细信息
# - recent_traces: 最近的溯源记录
```

### 4. 导出溯源信息到文件

```python
# 导出所有变量的溯源信息到JSON文件
env.export_nan_trace_to_file('/path/to/nan_trace.json')

# 导出特定变量的溯源信息
env.export_nan_trace_to_file('/path/to/reward_nan_trace.json', variable_name='reward')
```

## 溯源信息包含的字段

每个溯源记录包含以下信息：

- `step`: 发生的训练步数
- `variable_name`: 变量名称
- `has_nan`: 是否包含 NaN
- `has_inf`: 是否包含 Inf
- `nan_count`: NaN 的数量
- `inf_count`: Inf 的数量
- `nan_ratio`: NaN 比例
- `inf_ratio`: Inf 比例
- `shape`: 张量形状
- `dtype`: 数据类型
- `device`: 设备（CPU/GPU）
- `valid_min/max/mean/std`: 有效值的统计信息
- `caller_file`: 调用文件路径
- `caller_line`: 调用行号
- `caller_function`: 调用函数名
- `caller_code`: 调用代码行
- `total_occurrences`: 该变量总共出现的次数
- `timestamp`: 时间戳

## 使用示例

### 示例1: 训练后查看所有NaN问题

```python
# 训练完成后
env.print_nan_trace_summary()
```

### 示例2: 在训练循环中定期检查

```python
# 每1000步检查一次
if step % 1000 == 0:
    summary = env.get_nan_trace_summary()
    if summary['total_events'] > 0:
        print(f"警告：已检测到 {summary['total_events']} 次NaN/Inf事件")
        print(f"涉及变量: {', '.join(summary['variables_with_nan'])}")
        
        # 查看首次出现的变量
        for var_name, info in summary['first_occurrences'].items():
            print(f"{var_name}: 首次出现在步数 {info['first_step']}")
```

### 示例3: 导出并分析溯源信息

```python
# 训练完成后导出
env.export_nan_trace_to_file('./nan_analysis.json')

# 使用Python分析
import json
with open('./nan_analysis.json', 'r') as f:
    data = json.load(f)

# 找出最频繁出现NaN的变量
from collections import Counter
var_counts = Counter([trace['variable_name'] for trace in data['recent_traces']])
print("最频繁出现NaN的变量:", var_counts.most_common(5))
```

## 注意事项

1. 溯源系统会保留最近1000条记录，超过的会被自动清理
2. 每次检测到NaN/Inf时都会立即打印详细信息
3. 首次出现的变量会特别标注，帮助定位根本原因
4. 溯源信息包含完整的调用栈，可以精确定位问题代码

## 性能影响

- 溯源系统在 `debug_mode=True` 时启用
- 对性能的影响很小（主要是日志记录）
- 可以随时通过设置 `env.debug_mode = False` 来禁用

