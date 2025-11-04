# KL散度NaN问题修复指南

## 问题原因分析

KL散度变成NaN通常由以下原因导致：

1. **log_std 值过大**：导致 sigma = exp(log_std) 过大，使得概率密度计算溢出
2. **sigma 值过小**：接近0时，概率密度计算中的除法会导致数值不稳定
3. **观测值异常**：输入到网络的观测包含NaN/Inf，导致网络输出异常
4. **新旧策略差异过大**：策略更新过快，新旧策略的概率比计算不稳定

## 已实施的修复

### 1. 配置文件修复 (`rl_games_diff_drive_cfg.yaml`)

```yaml
space:
  continuous:
    sigma_activation: "softplus"  # ✅ 添加激活函数限制sigma范围
    sigma_init:
      val: -1.0  # ✅ 降低初始log_std（对应sigma ≈ 0.37）
    min_sigma: 1e-6   # ✅ 最小sigma值，防止除零
    max_sigma: 1.0    # ✅ 最大sigma值，防止过大探索
```

### 2. 观测数据保护（已在代码中实现）

- ✅ 每个obs组件都有独立的数值稳定性检查
- ✅ 所有组件都有合理的数值范围限制
- ✅ 最终obs有全局检查和回退机制

## 如果问题仍然存在

### 方案1：检查观测归一化

在配置文件中启用观测归一化：
```yaml
config:
  normalize_input: True  # 改为 True，对观测进行归一化
```

### 方案2：进一步降低初始探索

如果KL散度仍然NaN，可以进一步降低初始log_std：
```yaml
sigma_init:
  val: -2.0  # 对应sigma ≈ 0.14，更小的初始探索
```

### 方案3：增加KL散度阈值

如果训练初期KL散度波动大，可以临时增加阈值：
```yaml
kl_threshold: 0.05  # 从0.01增加到0.05
```

### 方案4：降低学习率

过大的学习率可能导致策略更新过快：
```yaml
learning_rate: 1e-5  # 从3e-5降低到1e-5
```

## 监控和诊断

### TensorBoard监控指标

查看以下指标判断问题：
- `Losses/kl` - KL散度值（应该 < kl_threshold）
- `Losses/loss` - 总损失（不应该有NaN）
- `Debug/obs_*_nan_inf_count` - 观测中的NaN/Inf计数

### 调试步骤

1. **检查观测是否有NaN/Inf**
   ```python
   # 训练时查看TensorBoard的Debug目录
   # 如果obs_*_nan_inf_count > 0，说明观测有问题
   ```

2. **检查策略网络输出**
   - KL散度NaN通常意味着mu或log_std输出异常
   - 查看网络权重是否有异常大的值

3. **检查动作分布**
   - 如果sigma过大，动作分布会过于分散
   - 如果sigma过小，探索不足

## 推荐配置组合

如果KL散度持续NaN，尝试以下配置：

```yaml
# 网络配置
network:
  space:
    continuous:
      sigma_activation: "softplus"
      sigma_init:
        val: -1.5  # 更保守的初始值
      min_sigma: 1e-6
      max_sigma: 0.5  # 更小的最大sigma

# 训练配置
config:
  normalize_input: True  # 启用观测归一化
  learning_rate: 1e-5    # 降低学习率
  kl_threshold: 0.02     # 适中的KL阈值
  grad_norm: 0.5         # 梯度裁剪
  e_clip: 0.15           # PPO裁剪范围
  mini_epochs: 2         # 减少更新次数
```

## 紧急修复

如果训练中突然出现KL散度NaN，可以：

1. **停止训练**
2. **检查最近的检查点**，看是否可以恢复
3. **降低学习率**后继续训练
4. **如果持续NaN，可能需要从头开始训练**

