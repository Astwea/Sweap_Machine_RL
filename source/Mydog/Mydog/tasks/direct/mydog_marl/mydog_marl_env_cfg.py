import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from .base_bot import MYDOG_CFG,SAODI_CONFIG
from isaaclab.terrains import TerrainImporterCfg


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )



@configclass
class MydogMarlEnvCfg(DirectRLEnvCfg):
    # 1. 环境参数配置
    # 1.1 基本环境参数
    # 由于机器人移动速度较慢（0.1 m/s），需要更长的回合时间来完成任务
    # 如果速度0.1 m/s，60秒可以移动6米，30秒可以移动3米
    episode_length_s = 60.0  # 每个回合的最大时长（秒）- 增加以适应慢速移动
    # 降低控制频率：速度慢时不需要太频繁的控制
    # decimation=40 意味着每40个仿真步（0.2秒）执行一次动作，控制频率约5Hz
    # 对于0.1 m/s的慢速移动，5Hz的控制频率已经足够
    decimation = 40  # 动作执行间隔（仿真步长倍数）- 控制频率5Hz (1/(0.005*40))
    action_scale = 1.0  # 线速度和角速度的缩放系数
    action_space = 2  # 动作空间维度 (线速度, 角速度)
    # 观测空间维度：线速度(2) + 角速度(1) + 动作(2) + 上一步动作(2) + 相对误差(2) + 轨迹差值(6) + 未来目标相对位置(6) + cos/sin(2) = 23
    observation_space = 23  # 观测空间维度（移除绝对轨迹点，只保留相对差值）
    state_space = 0  # 全局状态空间维度，0表示未定义
    wheel_base = 0.233  # 机器人轮距（米）
    # 2. 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # 每步仿真时间间隔（秒），仿真频率200Hz
        # 渲染间隔独立于控制频率，保持流畅的视觉效果
        # render_interval=4 意味着每4个仿真步渲染一次，渲染频率约50Hz（流畅）
        render_interval=4,  # 渲染间隔（每隔多少仿真步渲染一次），与decimation独立
    )
    # 3. 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # 控制频率降低到5Hz后，动作更新计算负担减少，可以支持更多环境
        # RTX 3060 6GB + 16GB内存：建议96个环境（如果显存不足可降回64）
        num_envs=96,  # 场景中的环境数量 - 从48增加到96，提高样本产出效率
        env_spacing=4.0,  # 每个环境的空间间隔（米）
        replicate_physics=True  # 是否复制物理属性
    )
    #events: EventCfg = EventCfg()
    # 4. 机器人配置
    # 4.1 机器人关节参数
    robot: ArticulationCfg = SAODI_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    tensorboard_dir = "/home/astwea/MyDogTask/Mydog/runs/logs"  # 日志目录
    num_waypoints = 5 # 路径点数量 - 增加轨迹复杂度
    num_interp = 12 # 插值数量 - 使轨迹更平滑
    step_size = 1.0  # 步长 - 减小提高精度
    # 5. 奖励缩放系数 - 优化后的权重，配合平滑奖励系统
    # - 用于平衡不同奖励项的相对重要性
    traj_track_scale = 15.0 # 增加路径跟踪权重
    traj_done_bonus = 1.0 # 写在 reward 里
    action_magnitude_scale = 0.1 # 增加动作幅度惩罚
    action_rate_reward_scale = 0.05 # 增加动作变化率惩罚
    direction_scale = 5.0 # 增加朝向奖励权重
    lateral_error_scale = 25.0 # 保持侧向误差惩罚
    imitation_scale = 10.0  # 模仿奖励缩放系数
    
    # 6. 课程学习配置
    enable_curriculum = True  # 是否启用课程学习
    # 课程切换的成功率阈值（每个阶段需要达到的最低成功率才能切换到下一阶段）
    curriculum_success_rate_threshold = 0.7  # 70%成功率阈值
    # 每个阶段的最小回合数（即使达到成功率也必须在当前阶段停留最小时长）
    curriculum_min_episodes_per_stage = [100, 200, 300]  # 各阶段最小回合数
    # 成功率评估窗口大小（评估最近N个回合的成功率）
    curriculum_success_window_size = 50  # 最近50个回合