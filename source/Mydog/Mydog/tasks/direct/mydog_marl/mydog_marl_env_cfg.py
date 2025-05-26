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
    episode_length_s = 20.0  # 每个回合的最大时长（秒）
    decimation = 2  # 动作执行间隔（仿真步长倍数）
    action_scale = 1.0  # 线速度和角速度的缩放系数
    action_space = 2  # 动作空间维度 (线速度, 角速度)
    observation_space = 21  # 观测空间维度 (x, y, yaw, 速度)
    state_space = 0  # 全局状态空间维度，0表示未定义
    wheel_base = 0.233  # 机器人轮距（米）
    # 2. 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # 每步仿真时间间隔（秒）
        render_interval=decimation,  # 渲染间隔（每隔多少仿真步渲染一次）
    )
    # 3. 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,  # 场景中的环境数量
        env_spacing=4.0,  # 每个环境的空间间隔（米）
        replicate_physics=True  # 是否复制物理属性
    )
    #events: EventCfg = EventCfg()
    # 4. 机器人配置
    # 4.1 机器人关节参数
    robot: ArticulationCfg = SAODI_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    log_dir = "runs/logs"  # 日志目录
    num_waypoints = 4  # 路径点数量
    num_interp = 10  # 插值数量
    step_size = 3.0  # 步长
    # 5. 奖励缩放系数
    # - 用于平衡不同奖励项的相对重要性
    lin_vel_reward_scale = 1 # 线速度跟踪奖励缩放系数
    ang_vel_reward_scale = 1 # 角速度跟踪奖励缩放系数
    action_rate_reward_scale = -1  # 动作变化惩罚缩放系数
    distance_scale = 50
    direction_scale = 50.0
    traj_progress_scale = 50.0
    still_penalty_scale = 5
    action_magnitude_scale = 0.2

