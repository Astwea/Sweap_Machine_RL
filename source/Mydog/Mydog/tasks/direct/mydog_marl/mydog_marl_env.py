import gymnasium as gym
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import spawn_ground_plane,spawn_from_usd, UsdFileCfg, GroundPlaneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.assets import AssetBaseCfg
from .mydog_marl_env_cfg import MydogMarlEnvCfg

try:
    from isaacsim.util.debug_draw import _debug_draw as debug_draw
    from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
    debug_draw_available = True
except ImportError:
    debug_draw_available = False
    print("Debug draw is not available. Please check your Isaac Sim installation.")

import math
from isaaclab.utils.math import quat_from_angle_axis
from torch.utils.tensorboard import SummaryWriter
import time
import traceback

import torch
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np
import random
from .mpc_controller import DifferentialDriveMPC
from concurrent.futures import ThreadPoolExecutor

def define_markers(path, idx) -> VisualizationMarkers:
    """Define markers with various different shapes."""
    if idx == 0:
        markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.5,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
            }
    elif idx == 1:
        markers={
                "arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                )
            }
    elif idx == 2:
        markers={
                "arrow1": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.5, 0.5, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
            }
    marker_cfg = VisualizationMarkersCfg(
        prim_path=f"/Visuals/myMarkers/{path}",
        markers=markers
    
    )
    return VisualizationMarkers(marker_cfg)

def mpc_worker(args):
    pos, yaw, start_idx, traj_np, horizon, step_dt = args
    mpc = DifferentialDriveMPC(horizon=horizon, dt=step_dt)
    sub_traj = traj_np[start_idx:start_idx + horizon + 1]
    if len(sub_traj) < horizon + 1:
        pad = np.tile(sub_traj[-1], (horizon + 1 - len(sub_traj), 1))
        sub_traj = np.concatenate([sub_traj, pad], axis=0)
    try:
        v, w = mpc.solve((pos[0], pos[1], yaw), sub_traj)
    except:
        v, w = 0.0, 0.0
    return [v, w]

class MydogMarlEnv(DirectRLEnv):
    # 1. 配置初始化
    cfg: MydogMarlEnvCfg

    def __init__(self, cfg: MydogMarlEnvCfg, render_mode: str | None = None, **kwargs):
        # 1.1 初始化父类
        super().__init__(cfg, render_mode, **kwargs)

        # 1.2 初始化动作存储
        # - 记录当前和前一次的动作，用于计算奖励和动态控制
        self._actions = torch.zeros(self.num_envs, 2, device=self.device)  # (线速度, 角速度)
        self.teacher_actions = torch.zeros(self.num_envs, 2, device=self.device)  # (线速度, 角速度)
        self._previous_actions = torch.zeros(self.num_envs, 2, device=self.device)


        self.arrow_visual = define_markers(path="arrows", idx=1)
        self.target_arrow_visual = define_markers(path="target_arrows", idx=2)

        self._commands = torch.zeros(self.num_envs, 2, device=self.device)  # (x, y, z)
        self._trajectories = torch.zeros(self.num_envs, self.cfg.num_waypoints*self.cfg.num_interp, 2, device=self.device)
        self._current_wp_idx = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.seed = 10
        self.epoch = 0
        self.turn_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.headerror = torch.zeros(self.num_envs, device=self.device)
        self.dist_to_target = None
        self._prev_dist_to_target = torch.zeros(self.num_envs, device=self.device)
        self.target_orientations = None
        self.positions = self._robot.data.root_state_w[:, :2]
        # 立即检查位置数据是否有NaN/Inf（在初始化时就检查）
        if isinstance(self.positions, torch.Tensor):
            if torch.isnan(self.positions).any() or torch.isinf(self.positions).any():
                print(f"🚨 警告：初始化时位置数据包含NaN/Inf！")
                self.positions = torch.where(torch.isnan(self.positions) | torch.isinf(self.positions),
                                           torch.zeros_like(self.positions), self.positions)
        self.last_pos = self.positions.clone()
        self.cos_phi = torch.zeros(self.num_envs, 1, device=self.device)
        self.sin_phi = torch.zeros(self.num_envs, 1, device=self.device)
        # 1.3 初始化日志记录
        # - 记录每个回合中的关键性能指标
        print(f"{cfg.tensorboard_dir}/{time.strftime('%Y-%m-%d_%H-%M-%S')}/summary ----------------------------------------------------------------")
        self.writer = SummaryWriter(log_dir=f"{cfg.tensorboard_dir}/{time.strftime('%Y-%m-%d_%H-%M-%S')}/summary")
        self.begin_time = time.time()
        self.global_step = 0
        self.epoch_step = 0
        self.episode_count = 0  # 回合结束次数计数器
        
        # 初始化各种指标记录
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) 
            for key in ["tracking_reward","direction_reward","goal_bias","action_rate_penalty","action_mag_penalty","imitation_reward"]
        }
        
        # 添加更多监控指标
        self._episode_metrics = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) 
            for key in ["episode_length", "success_rate", "total_distance", "avg_speed", "max_lateral_error", "final_distance"]
        }
        
        # 性能统计
        self._performance_stats = {
            "fps": 0.0,
            "step_time": 0.0,
            "reward_time": 0.0,
            "action_time": 0.0
        }
        
        # 训练统计
        self._training_stats = {
            "episode_count": 0,
            "total_episodes": 0,
            "best_reward": float('-inf'),
            "avg_reward": 0.0
        }
        self.joint_idx, _ = self._robot.find_joints(['left_wheel_joint', 'right_wheel_joint'])
        self.positions = self._robot.data.root_state_w[:, :2]
        # 注意：此时_always_check_nan可能还未初始化，使用简单检查
        if isinstance(self.positions, torch.Tensor):
            if torch.isnan(self.positions).any() or torch.isinf(self.positions).any():
                print(f"🚨 警告：初始化时位置数据包含NaN/Inf！")
                self.positions = torch.where(torch.isnan(self.positions) | torch.isinf(self.positions),
                                           torch.zeros_like(self.positions), self.positions)
        
        self.yaw = self._robot.data.root_state_w[:, 3:7]
        if isinstance(self.yaw, torch.Tensor):
            if torch.isnan(self.yaw).any() or torch.isinf(self.yaw).any():
                print(f"🚨 警告：初始化时yaw数据包含NaN/Inf！")
                self.yaw = torch.where(torch.isnan(self.yaw) | torch.isinf(self.yaw),
                                      torch.zeros_like(self.yaw), self.yaw)
        
        self.mpc = [DifferentialDriveMPC(horizon=10, dt=self.step_dt) for _ in range(self.num_envs)]
        
        # === 奖励平滑系统初始化 ===
        # 目标点切换检测
        self._prev_wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._target_switch_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 奖励平滑历史状态
        self._prev_rewards = {
            'heading_reward': None,
            'progress_reward': None,
            'lateral_penalty': None,
            'goal_reward': None,
        }
        
        # 平滑参数 - 优化后的参数
        self.smoothing_factor = 0.15  # 平滑因子 - 增加平滑效果
        self.transition_smoothing_factor = 0.4  # 目标点切换时的平滑因子 - 增强切换平滑
        
        # 数值稳定性检查 - 必须在所有初始化之前设置
        self.debug_mode = True  # 启用调试模式
        self.nan_inf_count = {'obs': 0, 'reward': 0, 'action': 0, 'quat': 0, 'trajectory': 0, 'position': 0, 'tensorboard': 0}
        
        # NaN/Inf 溯源系统
        self.nan_trace_log = []  # 记录所有 NaN/Inf 事件的详细日志
        self.nan_first_occurrence = {}  # 记录每个变量第一次出现 NaN 的位置
        
        # 全局NaN检测开关 - 始终启用，即使在初始化阶段
        self._always_check_nan = True  # 强制启用NaN检测，即使debug_mode关闭
        
        # === 课程学习系统初始化 ===
        self.curriculum_enabled = self.cfg.enable_curriculum
        self.curriculum_stage = 0  # 当前课程阶段
        
        # 定义课程学习阶段（基于成功率和最小回合数控制切换）
        # 注意：由于机器人移动速度慢（0.1 m/s），回合长度已相应增加
        # 原配置：10s, 12s, 15s → 新配置：40s, 50s, 60s（约4倍比例）
        # 这样机器人有足够时间完成任务，避免梯度消失问题
        self.curriculum_stages = {
            0: {  # 阶段1: 基础直线跟踪
                'num_waypoints': 2,
                'num_interp': 4,
                'step_size': 0.5,
                'episode_length_s': 40.0,  # 从10秒增加到40秒，适应慢速移动（0.1 m/s）
                'traj_track_scale': 20.0,
                'lateral_error_scale': 30.0,
                'direction_scale': 8.0,
                'stage_name': 'basic_straight'
            },
            1: {  # 阶段2: 简单转弯
                'num_waypoints': 3,
                'num_interp': 6,
                'step_size': 0.8,
                'episode_length_s': 50.0,  # 从12秒增加到50秒，适应慢速移动
                'traj_track_scale': 15.0,
                'lateral_error_scale': 25.0,
                'direction_scale': 6.0,
                'stage_name': 'simple_turn'
            },
            2: {  # 阶段3: 复杂轨迹
                'num_waypoints': 5,
                'num_interp': 12,
                'step_size': 1.0,
                'episode_length_s': 60.0,  # 从15秒增加到60秒，与主配置一致
                'traj_track_scale': 15.0,
                'lateral_error_scale': 25.0,
                'direction_scale': 5.0,
                'stage_name': 'complex_trajectory'
            }
        }
        # 课程学习配置参数
        self.curriculum_success_rate_threshold = cfg.curriculum_success_rate_threshold
        self.curriculum_min_episodes_per_stage = cfg.curriculum_min_episodes_per_stage
        self.curriculum_success_window_size = cfg.curriculum_success_window_size
        
        # 课程学习统计
        self.curriculum_stats = {
            'stage_0_steps': 0,
            'stage_1_steps': 0,
            'stage_2_steps': 0,
            'stage_0_success_rate': 0.0,
            'stage_1_success_rate': 0.0,
            'stage_2_success_rate': 0.0,
        }
        
        # 成功率历史记录（用于滑动窗口评估）
        self.success_history = []  # 记录最近的成功/失败历史
        
        # 初始化完成后检查位置数据
        self.positions = self._check_numerical_stability(self.positions, 'position')
        self.last_pos = self._check_numerical_stability(self.last_pos, 'position')
        
        # 初始化课程学习参数
        self._init_curriculum_parameters()

        if debug_draw_available:
            self.debug_draw = debug_draw.acquire_debug_draw_interface()
            keyboard_cfg = Se2KeyboardCfg(v_y_sensitivity=0.8)
            self.keyboard = Se2Keyboard(keyboard_cfg)
        else:
            self.debug_draw = None
            self.keyboard = None
        self.count = 0
        self.finished_mask = None
        
    # 2. 场景设置
    def _setup_scene(self):
        # 2.1 初始化机器人模型
        self._robot = Articulation(self.cfg.robot)
        # spawn_ground_plane(prim_path="/World/ground", cfg=Envconfig())
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(color=(0.5, 0.5, 0.5),
                               size=(300, 300))
        )
        self.scene.articulations["robot"] = self._robot
        
        # 2.2 初始化接触传感器
        # self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # self.scene.sensors["contact_sensor"] = self._contact_sensor
        # self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        # self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # 2.3 克隆环境
        # - 创建多个环境实例，提高并行效率
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    # 3. 物理步前处理

    def quaternion_to_forward_vector(self, quaternions):
        # 提取四元数 (qx, qy, qz, qw)
        qw, qx, qy, qz = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

        # 计算正前方方向向量
        forward_x = 2 * (qx * qz + qw * qy)
        forward_y = 2 * (qy * qz - qw * qx)
        forward_z = 1 - 2 * (qx ** 2 + qy ** 2)

        return torch.stack([forward_x, forward_y, forward_z], dim=1)


    def quaternion_to_yaw(self,quat):
        """
        计算四元数 (w, x, y, z) 对应的 yaw 角度
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # 数值稳定性检查
        quat = self._check_numerical_stability(quat, 'quat')
        
        # 确保四元数归一化，防止除零和sqrt负数
        # 先检查输入是否有NaN/Inf
        w, x, y, z = self._check_numerical_stability(w, 'quat_w_before_norm'), \
                     self._check_numerical_stability(x, 'quat_x_before_norm'), \
                     self._check_numerical_stability(y, 'quat_y_before_norm'), \
                     self._check_numerical_stability(z, 'quat_z_before_norm')
        
        # 计算norm，确保输入非负（防止sqrt负数产生NaN）
        norm_sq = w**2 + x**2 + y**2 + z**2
        norm_sq = torch.clamp(norm_sq, min=0.0)  # 确保非负
        norm = torch.sqrt(norm_sq + 1e-10)  # 添加小值防止sqrt(0)的数值问题
        norm = torch.clamp(norm, min=1e-8)  # 避免除零，确保最小值
        # 检查norm是否有异常值
        norm = self._check_numerical_stability(norm, 'quat_norm')
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        # 检查归一化后的四元数
        w, x, y, z = self._check_numerical_stability(w, 'quat_w'), \
                     self._check_numerical_stability(x, 'quat_x'), \
                     self._check_numerical_stability(y, 'quat_y'), \
                     self._check_numerical_stability(z, 'quat_z')
        
        # 计算atan2前检查输入是否有NaN/Inf
        numerator = 2 * (w * z + x * y)
        denominator = 1 - 2 * (y**2 + z**2)
        numerator = self._check_numerical_stability(numerator, 'atan2_numerator')
        denominator = self._check_numerical_stability(denominator, 'atan2_denominator')
        
        yaw = torch.atan2(numerator, denominator)
        
        # 检查结果并修复
        yaw = self._check_numerical_stability(yaw, 'yaw_result')
        
        return yaw

    def compute_orientation(self, pos, target):
        # 检查输入数值稳定性
        pos = self._check_numerical_stability(pos, 'compute_orientation_pos')
        target = self._check_numerical_stability(target, 'compute_orientation_target')
        
        # 计算目标方向
        direction_to_target = target - pos
        direction_to_target = self._check_numerical_stability(direction_to_target, 'direction_to_target')
        
        # 防止除零或异常值
        direction_to_target = torch.clamp(direction_to_target, -1000.0, 1000.0)
        
        yaw_target = torch.atan2(direction_to_target[:, 1], direction_to_target[:, 0])
        yaw_target = self._check_numerical_stability(yaw_target, 'yaw_target')
        
        # 如果仍有NaN，替换为零
        yaw_target = torch.where(torch.isnan(yaw_target) | torch.isinf(yaw_target), 
                                 torch.zeros_like(yaw_target), yaw_target)
        
        qx = torch.zeros_like(yaw_target)
        qy = torch.zeros_like(yaw_target)
        qz = torch.sin(yaw_target / 2)
        qw = torch.cos(yaw_target / 2)
        
        # 检查四元数组件
        qz = self._check_numerical_stability(qz, 'qz')
        qw = self._check_numerical_stability(qw, 'qw')
        
        quat = torch.stack([qw, qx, qy, qz], dim=1)
        quat = self._check_numerical_stability(quat, 'compute_orientation_quat')
        
        return quat
    

    def get_teacher_action(self):
        poses = self._robot.data.root_state_w[:, :2].cpu().numpy()
        yaws = self.quaternion_to_yaw(self._robot.data.root_state_w[:, 3:7]).cpu().numpy()
        idxs = self._current_wp_idx.cpu().numpy()
        trajs = self._trajectories.detach().cpu().numpy()
        horizon = 10

        # 把 step_dt 加到每个args
        args_list = [
            (poses[i], yaws[i], idxs[i], trajs[i], horizon, self.step_dt)
            for i in range(self.num_envs)
        ]
        with ThreadPoolExecutor() as executor:
            actions = list(executor.map(mpc_worker, args_list))

        return torch.tensor(actions, dtype=torch.float32, device=self.device)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        # 3.1 缓存当前动作
        # - 记录输入的动作，以便后续使用
        # action = self.keyboard.advance()
        # vx, wz = action[0], action[1]
        # self._actions = torch.tensor(np.tile([vx, wz], (self.num_envs, 1)), dtype=torch.float32, device=self.device)
        # v, w = self.mpc.solve(
        #     init_state=(x, y, yaw),
        #     ref_traj=your_traj_np_array  # 轨迹为 np.array([[x0, y0], [x1, y1], ..., [xN, yN]])
        # )
        #self.teacher_actions = self.get_teacher_action()   # [N,2] tensor
        #
        # 立即检查输入动作是否有NaN/Inf（最优先检查，防止NaN从外部传入）
        if isinstance(actions, torch.Tensor):
            if torch.isnan(actions).any() or torch.isinf(actions).any():
                print(f"🚨 严重警告：输入动作包含NaN/Inf！步数：{getattr(self, 'global_step', 0)}")
                print(f"   NaN数量: {torch.isnan(actions).sum().item()}, Inf数量: {torch.isinf(actions).sum().item()}")
                # 立即修复
                actions = torch.where(torch.isnan(actions) | torch.isinf(actions), 
                                    torch.zeros_like(actions), actions)
        
        # 使用统一的数值稳定性检查
        actions = self._check_numerical_stability(actions, 'action_input')
        
        # 动作已经在模型内部通过tanh限制在[-1, 1]
        # 这里应用缩放因子，将动作映射到实际速度范围
        # 使用乘法缩放（有梯度），而不是clamp（无梯度）
        # 模型输出：tanh输出[-1,1] * scale = 实际动作范围
        max_linear_vel = 0.1  # m/s - 实际最大线速度
        max_angular_vel = 1.0  # rad/s - 实际最大角速度
        
        # 应用缩放（乘法操作有梯度，可以反向传播到模型）
        scaled_actions = torch.stack([
            actions[:, 0] * max_linear_vel,   # 线速度：[-1,1] * 0.1 = [-0.1, 0.1] m/s
            actions[:, 1] * max_angular_vel   # 角速度：[-1,1] * 1.0 = [-1.0, 1.0] rad/s
        ], dim=1)
        
        self._actions = scaled_actions
        
        # 数值稳定性检查（不限制范围，只检查NaN/Inf）
        self._actions = self._check_numerical_stability(self._actions, 'final_actions')

    def adjust_yaw_with_velocity_tensor(self, quat, vx):
        # 提取四元数分量
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # 计算当前的 yaw
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        
        # 修正 yaw
        reversing_mask = vx < 0
        yaw[reversing_mask] = (yaw[reversing_mask] + math.pi) % (2 * math.pi)

        # 计算新的四元数
        half_yaw = yaw / 2
        sin_half_yaw = torch.sin(half_yaw)
        cos_half_yaw = torch.cos(half_yaw)

        # 直接修改原始四元数中的 z 和 w
        adjusted_quat = quat.clone()
        adjusted_quat[:, 3] = sin_half_yaw
        adjusted_quat[:, 0] = cos_half_yaw

        return adjusted_quat

    # 4. 应用动作到物理引擎
    def _apply_action(self):
        # 4.1 将动作映射为左右轮速度
        linear_vel, angular_vel = self._actions[:, 0], self._actions[:, 1]
        
        # 检查动作数值稳定性
        linear_vel = self._check_numerical_stability(linear_vel, 'action_linear_vel')
        angular_vel = self._check_numerical_stability(angular_vel, 'action_angular_vel')
        
        # 确保动作在合理范围内
        linear_vel = torch.clamp(linear_vel, -1.0, 1.0)
        angular_vel = torch.clamp(angular_vel, -2.0, 2.0)
        
        # # 4.2 计算左右轮速度
        left_wheel_vel = linear_vel - angular_vel * self.cfg.wheel_base / 2
        right_wheel_vel = linear_vel + angular_vel * self.cfg.wheel_base / 2
        
        # # 4.3 设置机器人关节速度目标
        wheel_radius = 0.0357  # 假设轮子半径是 5cm
        # 防止除以零或很小的值（使用张量操作确保对所有环境都安全）
        wheel_radius_tensor = torch.tensor(wheel_radius, device=self.device, dtype=torch.float32)
        wheel_radius_tensor = torch.clamp(wheel_radius_tensor, min=1e-6)  # 确保不为零
        
        left_wheel_vel = left_wheel_vel / wheel_radius_tensor
        right_wheel_vel = right_wheel_vel / wheel_radius_tensor
        
        # 检查计算后的速度是否有异常值
        left_wheel_vel = self._check_numerical_stability(left_wheel_vel, 'left_wheel_vel')
        right_wheel_vel = self._check_numerical_stability(right_wheel_vel, 'right_wheel_vel')
        
        # 限制轮子速度在合理范围内（对应速度限制3.0 rad/s）
        left_wheel_vel = torch.clamp(left_wheel_vel, -3.0, 3.0)
        right_wheel_vel = torch.clamp(right_wheel_vel, -3.0, 3.0)
        
        # # 只选择左右轮的关节
        #    # 设置关节速度目标
        # zero_joint_vel = torch.zeros_like(left_wheel_vel)
        joint_vels = torch.stack([left_wheel_vel, right_wheel_vel],dim=1)
        
        # 最终检查
        joint_vels = self._check_numerical_stability(joint_vels, 'joint_vels')
        
        # 只设置左右轮的速度
        self._robot.set_joint_velocity_target(joint_vels, joint_ids=self.joint_idx)
        
        # 更新位置和yaw，并立即检查
        self.positions = self._robot.data.root_state_w[:, :2]
        self.positions = self._check_numerical_stability(self.positions, 'positions_after_physics')
        
        self.yaw = self._robot.data.root_state_w[:, 3:7]
        self.yaw = self._check_numerical_stability(self.yaw, 'yaw_after_physics')
        # 检查positions是否有NaN/Inf（在转换为tensor前）
        self.positions = self._check_numerical_stability(self.positions, 'positions_before_tensor_conversion')
        
        positions = torch.tensor([[x, y, 0.5] for x, y in self.positions], device=self.device)
        positions = self._check_numerical_stability(positions, 'positions_tensor')
        
        # 检查轨迹数据
        current_traj_indices = self._trajectories[range(self.num_envs), self._current_wp_idx]
        current_traj_indices = self._check_numerical_stability(current_traj_indices, 'current_traj_for_orientation')
        
        self.target_orientations = self.compute_orientation(self.positions, current_traj_indices)
        self.target_orientations = self._check_numerical_stability(self.target_orientations, 'target_orientations')
        
        # 计算角度差
        target_yaw = self.quaternion_to_yaw(self.target_orientations)
        robot_yaw = self.quaternion_to_yaw(self.yaw)
        angle_diff = target_yaw - robot_yaw
        angle_diff = self._check_numerical_stability(angle_diff, 'angle_diff')
        
        self.cos_phi = torch.cos(angle_diff).unsqueeze(1)
        self.sin_phi = torch.sin(angle_diff).unsqueeze(1)
        self.cos_phi = self._check_numerical_stability(self.cos_phi, 'cos_phi')
        self.sin_phi = self._check_numerical_stability(self.sin_phi, 'sin_phi')
        linear_vel = self._robot.data.root_lin_vel_b[:, 0]
        self.arrow_visual.visualize(translations=positions, orientations=self.yaw, marker_indices=torch.zeros(self.num_envs, dtype=torch.int64))
        self.target_arrow_visual.visualize(translations=positions, orientations=self.target_orientations, marker_indices=torch.zeros(self.num_envs, dtype=torch.int64))



    # 5. 获取观测数据
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        future_len = 4  # 当前点 + 未来3个轨迹点
        traj_points = []
        traj_deltas = []
        traj_len = self._trajectories.shape[1]
        # 检查positions是否有异常值（在计算距离前）
        self.positions = self._check_numerical_stability(self.positions, 'obs_positions')
        
        next_idx = torch.clamp(self._current_wp_idx + 1, max=traj_len - 1)
        # 检查轨迹数据是否有异常值
        current_traj = self._trajectories[range(self.num_envs), self._current_wp_idx]
        next_traj = self._trajectories[range(self.num_envs), next_idx]
        current_traj = self._check_numerical_stability(current_traj, 'obs_current_traj')
        next_traj = self._check_numerical_stability(next_traj, 'obs_next_traj')
        
        # 计算距离前确保数值稳定
        pos_curr_diff = self.positions - current_traj
        pos_next_diff = self.positions - next_traj
        
        pos_curr_diff = self._check_numerical_stability(pos_curr_diff, 'pos_curr_diff')
        pos_next_diff = self._check_numerical_stability(pos_next_diff, 'pos_next_diff')
        
        # 在计算norm前检查输入是否有NaN/Inf
        pos_curr_diff = self._check_numerical_stability(pos_curr_diff, 'pos_curr_diff_before_norm')
        pos_next_diff = self._check_numerical_stability(pos_next_diff, 'pos_next_diff_before_norm')
        
        dist_curr = torch.norm(pos_curr_diff, dim=1)
        dist_next = torch.norm(pos_next_diff, dim=1)
        
        # 检查距离计算是否有异常
        dist_curr = self._check_numerical_stability(dist_curr, 'dist_curr')
        dist_next = self._check_numerical_stability(dist_next, 'dist_next')
        
        # 限制距离范围，防止异常值
        dist_curr = torch.clamp(dist_curr, 0.0, 1000.0)
        dist_next = torch.clamp(dist_next, 0.0, 1000.0)

        # 如果离下一个更近，就推进 index
        advance_condition = (dist_next + 0.05 < dist_curr) & (dist_curr < 0.4)

        self._current_wp_idx = torch.where(
            advance_condition,
            torch.clamp(self._current_wp_idx + 1, max=traj_len - 1),
            self._current_wp_idx
        )
        for i in range(future_len):
            idx = torch.clamp(self._current_wp_idx + i, max=self._trajectories.shape[1] - 1)
            point = self._trajectories[torch.arange(self.num_envs), idx]
            # 检查轨迹点是否有异常值
            point = self._check_numerical_stability(point, f'obs_traj_point_{i}')
            point = torch.clamp(point, -1000.0, 1000.0)  # 限制轨迹点范围
            traj_points.append(point)
            if i > 0:
                prev_idx = torch.clamp(self._current_wp_idx + i - 1, max=self._trajectories.shape[1] - 1)
                prev_point = self._trajectories[torch.arange(self.num_envs), prev_idx]
                prev_point = self._check_numerical_stability(prev_point, f'obs_traj_prev_point_{i}')
                delta = point - prev_point
                delta = self._check_numerical_stability(delta, f'obs_traj_delta_{i}')
                traj_deltas.append(delta)
        
        # 轨迹差值特征（相邻轨迹点之间的相对差值，表示轨迹趋势）
        traj_delta_feats = torch.cat(traj_deltas, dim=-1) # [num_envs, 2 * (future_len - 1)]

        # 当前目标点相对于机器人的位置（目标相对于机器人的方向）
        # 注意：应该是 target - pos，这样向量指向目标，机器人知道目标在哪个方向
        current_target = traj_points[0]
        current_target = self._check_numerical_stability(current_target, 'obs_current_target')
        current_pos = self._check_numerical_stability(self.positions, 'obs_current_pos_for_error')
        
        relative_error = current_target - current_pos  # [num_envs, 2] 指向目标的方向
        relative_error = self._check_numerical_stability(relative_error, 'obs_relative_error_calc')
        
        # 未来目标点相对于当前目标点的位置（轨迹前进方向）
        # 这样机器人可以知道轨迹的延伸方向，而不仅仅是当前目标点
        future_targets_relative = []
        for i in range(1, min(future_len, len(traj_points))):
            future_target = traj_points[i]
            future_target = self._check_numerical_stability(future_target, f'obs_future_target_{i}')
            # 未来目标点相对于当前目标点的位置
            future_relative = future_target - current_target
            future_relative = self._check_numerical_stability(future_relative, f'obs_future_relative_{i}')
            future_targets_relative.append(future_relative)
        
        if future_targets_relative:
            future_targets_relative = torch.cat(future_targets_relative, dim=-1)  # [num_envs, 2 * (future_len - 1)]
        else:
            future_targets_relative = torch.zeros(self.num_envs, 0, device=self.device)

        # yaw 方向 → cos/sin(yaw)
        # 检查yaw数值稳定性
        yaw_check = self._check_numerical_stability(self.yaw, 'obs_yaw_quat')
        yaw_tensor = self.quaternion_to_yaw(yaw_check)
        yaw_tensor = self._check_numerical_stability(yaw_tensor, 'obs_yaw_tensor')
        
        # 限制yaw角度范围
        yaw_tensor = torch.clamp(yaw_tensor, -10.0, 10.0)
        
        cos_yaw = torch.cos(yaw_tensor).unsqueeze(1)
        sin_yaw = torch.sin(yaw_tensor).unsqueeze(1)
        
        # 检查cos/sin计算结果
        cos_yaw = self._check_numerical_stability(cos_yaw, 'obs_cos_yaw_calc')
        sin_yaw = self._check_numerical_stability(sin_yaw, 'obs_sin_yaw_calc')

        # === 分别检查每个obs组件的数值稳定性 ===
        # 检查各个输入组件，防止INF传播
        lin_vel = self._robot.data.root_lin_vel_b[:, :2]
        lin_vel = self._check_numerical_stability(lin_vel, 'obs_lin_vel')
        lin_vel = torch.clamp(lin_vel, -100.0, 100.0)  # 限制速度范围
        
        ang_vel = self._robot.data.root_ang_vel_b[:, 2:]
        ang_vel = self._check_numerical_stability(ang_vel, 'obs_ang_vel')
        ang_vel = torch.clamp(ang_vel, -10.0, 10.0)  # 限制角速度范围
        
        actions = self._check_numerical_stability(self._actions, 'obs_actions')
        prev_actions = self._check_numerical_stability(self._previous_actions, 'obs_prev_actions')
        
        relative_error = self._check_numerical_stability(relative_error, 'obs_relative_error')
        relative_error = torch.clamp(relative_error, -100.0, 100.0)  # 限制位置误差范围
        
        # 移除绝对轨迹点，只保留相对差值
        # traj_feats 不再需要，因为机器人只需要知道相对位置
        
        traj_delta_feats = self._check_numerical_stability(traj_delta_feats, 'obs_traj_delta_feats')
        traj_delta_feats = torch.clamp(traj_delta_feats, -100.0, 100.0)  # 限制轨迹差值范围
        
        # 未来目标点相对于当前目标点的位置
        if future_targets_relative.numel() > 0:
            future_targets_relative = self._check_numerical_stability(future_targets_relative, 'obs_future_targets_relative')
            future_targets_relative = torch.clamp(future_targets_relative, -100.0, 100.0)  # 限制范围
        else:
            future_targets_relative = torch.zeros(self.num_envs, 0, device=self.device)
        
        cos_yaw = self._check_numerical_stability(cos_yaw, 'obs_cos_yaw')
        sin_yaw = self._check_numerical_stability(sin_yaw, 'obs_sin_yaw')
        # cos和sin理论上应该在[-1,1]范围内，但为了安全也clamp一下
        cos_yaw = torch.clamp(cos_yaw, -1.0, 1.0)
        sin_yaw = torch.clamp(sin_yaw, -1.0, 1.0)

        # === 观测归一化：将各个组件缩放到相近的范围，提高归一化效果 ===
        # 1. 线速度：归一化到[-1, 1]范围（假设最大速度0.1 m/s）
        lin_vel_normalized = lin_vel / 0.1  # 归一化：[-0.1, 0.1] -> [-1, 1]
        lin_vel_normalized = torch.clamp(lin_vel_normalized, -1.0, 1.0)
        
        # 2. 角速度：归一化到[-1, 1]范围（假设最大角速度1.0 rad/s）
        ang_vel_normalized = ang_vel / 1.0  # 归一化：[-1, 1] -> [-1, 1]（已经是归一化的）
        ang_vel_normalized = torch.clamp(ang_vel_normalized, -1.0, 1.0)
        
        # 3. 动作：已经是归一化的（线速度约[-0.1, 0.1]，角速度约[-1, 1]）
        actions_normalized = torch.stack([
            actions[:, 0] / 0.1,   # 线速度归一化：[-0.1, 0.1] -> [-1, 1]
            actions[:, 1] / 1.0    # 角速度归一化：[-1, 1] -> [-1, 1]
        ], dim=1)
        actions_normalized = torch.clamp(actions_normalized, -1.0, 1.0)
        
        prev_actions_normalized = torch.stack([
            prev_actions[:, 0] / 0.1,   # 线速度归一化
            prev_actions[:, 1] / 1.0     # 角速度归一化
        ], dim=1)
        prev_actions_normalized = torch.clamp(prev_actions_normalized, -1.0, 1.0)
        
        # 4. 相对误差：归一化到[-1, 1]范围（假设最大误差10米）
        relative_error_normalized = relative_error / 10.0  # 归一化：[-10, 10] -> [-1, 1]
        relative_error_normalized = torch.clamp(relative_error_normalized, -1.0, 1.0)
        
        # 5. 轨迹差值：归一化到[-1, 1]范围（假设最大差值1米）
        traj_delta_feats_normalized = traj_delta_feats / 1.0  # 归一化：[-1, 1] -> [-1, 1]
        traj_delta_feats_normalized = torch.clamp(traj_delta_feats_normalized, -1.0, 1.0)
        
        # 6. 未来目标点相对位置：归一化到[-1, 1]范围（假设最大差值1米）
        if future_targets_relative.numel() > 0:
            future_targets_relative_normalized = future_targets_relative / 1.0  # 归一化：[-1, 1] -> [-1, 1]
            future_targets_relative_normalized = torch.clamp(future_targets_relative_normalized, -1.0, 1.0)
        else:
            future_targets_relative_normalized = torch.zeros(self.num_envs, 0, device=self.device)
        
        # 7. cos/sin：已经是[-1, 1]范围，无需归一化

        obs = torch.cat([
            lin_vel_normalized,                    # 线速度 (2,) - 归一化到[-1, 1]
            ang_vel_normalized,                   # 角速度 (1,) - 归一化到[-1, 1]
            actions_normalized,                   # 当前动作 (2,) - 归一化到[-1, 1]
            prev_actions_normalized,              # 上一步动作 (2,) - 归一化到[-1, 1]
            relative_error_normalized,            # 当前目标相对位置 (2,) - 归一化到[-1, 1]
            traj_delta_feats_normalized,          # 轨迹趋势（相邻点差值）(2×3) - 归一化到[-1, 1]
            future_targets_relative_normalized,    # 未来目标相对位置 (2×3) - 归一化到[-1, 1]
            cos_yaw, sin_yaw                      # 姿态信息 (2,) - 已经是[-1, 1]
        ], dim=-1)
        
        # 使用统一的数值稳定性检查（最终检查）
        obs = self._check_numerical_stability(obs, 'obs')
        
        # 归一化后的观测值应该在[-1, 1]范围内，但为了安全也clamp一下
        obs = torch.clamp(obs, -1.0, 1.0)  # 限制在[-1, 1]范围内
        
        # 强制替换所有NaN和Inf（双重保险）
        obs = torch.where(torch.isnan(obs) | torch.isinf(obs), torch.zeros_like(obs), obs)
        
        # 额外的安全检查
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"严重警告：归一化后的观测数据中仍有NaN/Inf，步数：{self.global_step}")
            print(f"尝试单独检查各个组件...")
            components = {
                'lin_vel_normalized': lin_vel_normalized,
                'ang_vel_normalized': ang_vel_normalized,
                'actions_normalized': actions_normalized,
                'prev_actions_normalized': prev_actions_normalized,
                'relative_error_normalized': relative_error_normalized,
                'traj_delta_feats_normalized': traj_delta_feats_normalized,
                'future_targets_relative_normalized': future_targets_relative_normalized if future_targets_relative_normalized.numel() > 0 else torch.zeros(1, 0, device=self.device),
                'cos_yaw': cos_yaw,
                'sin_yaw': sin_yaw
            }
            for name, comp in components.items():
                if torch.isnan(comp).any() or torch.isinf(comp).any():
                    print(f"  ⚠️  发现 {name} 包含 NaN/Inf!")
            # 如果仍有NaN/Inf，强制替换为零
            obs = torch.where(torch.isnan(obs) | torch.isinf(obs), torch.zeros_like(obs), obs)
        
        # 最终验证：确保没有任何NaN/Inf（如果仍有，强制替换为零）
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"⚠️  最终清理：强制替换归一化观测中剩余的NaN/Inf，步数：{self.global_step}")
            obs = torch.where(torch.isnan(obs) | torch.isinf(obs), torch.zeros_like(obs), obs)
        
        # 最终范围限制：确保归一化后的观测值在[-1, 1]范围内
        obs = torch.clamp(obs, -1.0, 1.0)  # 归一化后的观测应该在[-1, 1]范围内
        
        # 最后一次强制NaN/Inf清理
        obs = torch.where(torch.isnan(obs) | torch.isinf(obs), torch.zeros_like(obs), obs)
        
        return {"policy": obs}

    # 6. 计算奖励
    def _get_rewards(self) -> torch.Tensor:
        # 性能监控开始
        step_start_time = time.time()
        self.global_step += 1

        # === 课程学习更新 ===
        if self.curriculum_enabled:
            self._update_curriculum_stage()

        # === 检测目标点切换 ===
        target_switched = self._current_wp_idx != self._prev_wp_idx
        self._target_switch_detected = target_switched
        self._prev_wp_idx = self._current_wp_idx.clone()

        # === 当前轨迹段 ===
        id = torch.clamp(self._current_wp_idx, max=self._trajectories.shape[1] - 2)
        current_target = self._trajectories[torch.arange(self.num_envs), id]
        next_target = self._trajectories[torch.arange(self.num_envs), id + 1]
        
        # === 向量定义 ===
        pos = self.positions
        vel = self._robot.data.root_lin_vel_w[:, :2]
        ab = next_target - current_target  # 路径方向向量
        pa = current_target - pos  # 从位置指向目标的方向（修正：应该是 target - pos，指向目标）
        # 计算路径方向向量的归一化，防止除零
        # 先检查输入是否有NaN/Inf
        ab = self._check_numerical_stability(ab, 'ab_before_norm')
        ab_norm = torch.norm(ab, dim=1, keepdim=True)
        ab_norm = self._check_numerical_stability(ab_norm, 'ab_norm_before_clamp')
        ab_norm = torch.clamp(ab_norm, min=1e-6)  # 确保不为零
        ab_unit = ab / ab_norm
        ab_unit = self._check_numerical_stability(ab_unit, 'ab_unit')

        # === 投影点（路径上的最近点） ===
        # 注意：ap 用于投影计算（从目标到位置），但用于方向计算应该用 pa
        ap = pos - current_target  # 从目标到位置（用于距离计算）
        t = torch.clamp(torch.sum(ap * ab, dim=1) / (torch.sum(ab * ab, dim=1) + 1e-6), 0.0, 1.0).unsqueeze(1)
        proj = current_target + t * ab

        # === 距离误差 ===
        # 在计算norm前检查输入
        pos_proj_diff = pos - proj
        pos_proj_diff = self._check_numerical_stability(pos_proj_diff, 'pos_proj_diff_before_norm')
        lateral_error = torch.norm(pos_proj_diff, dim=1)
        lateral_error = self._check_numerical_stability(lateral_error, 'lateral_error')
        
        # 检查pa并计算距离
        pa = self._check_numerical_stability(pa, 'pa_before_norm')
        dist_to_target = torch.norm(pa, dim=1)  # 使用指向目标的方向计算距离
        dist_to_target = self._check_numerical_stability(dist_to_target, 'dist_to_target')
        self.dist_to_target = dist_to_target  # 用于 bias 计算

        # === 路径推进奖励（路径切向速度） ===
        forward_vector = self.quaternion_to_yaw(self.yaw)
        heading = torch.stack([torch.cos(forward_vector), torch.sin(forward_vector)], dim=1)  # shape: [N, 2]
        v_forward = torch.sum(vel * ab_unit, dim=1)
        alignment = torch.sum(heading * ab_unit, dim=1)  # cos(heading_angle - path_angle)
        # 修正：只有当速度方向和路径方向一致（alignment > 0）且向前移动（v_forward > 0）时才给奖励
        progress_reward = torch.tanh(v_forward) * torch.clamp(alignment, min=0.0)  # 只奖励朝向正确方向的移动

        # === 到达奖励 ===
        done_mask = (self._current_wp_idx >= self._trajectories.shape[1] - 1) & (dist_to_target < 0.2)
        bias = torch.zeros_like(dist_to_target)
        mask = (dist_to_target >= 0.2) & (dist_to_target < 0.5)
        bias[mask] = 0.0
        bias[done_mask] = 0.0  # 终点 bonus

        # === 朝向奖励（改进版 - 平滑处理） ===
        # 修正：target_heading 应该是朝向目标的方向，即 pa 的角度（target - pos）
        target_heading = torch.atan2(pa[:, 1], pa[:, 0])  # 从位置指向目标的方向
        next_heading = torch.atan2(ab[:, 1], ab[:, 0])  # 路径方向
        heading_error = forward_vector - target_heading  # 机器人朝向与目标方向的误差
        
        # 将角度误差标准化到 [-π, π]
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
        self.headerror = heading_error

        # 使用高斯函数替代线性惩罚，更平滑
        # 注意：heading_error 越小（接近0），奖励越大，鼓励机器人朝向目标
        direction_reward = torch.exp(-heading_error.abs() * 2.0)  # 高斯形式，更平滑
        
        # 添加调试信息：记录方向相关信息（每100步记录一次）
        if self.global_step % 100 == 0 and hasattr(self, 'writer'):
            self._safe_add_scalar("Debug/Target_Heading_Deg", torch.rad2deg(target_heading).mean(), self.global_step)
            self._safe_add_scalar("Debug/Robot_Heading_Deg", torch.rad2deg(forward_vector).mean(), self.global_step)
            self._safe_add_scalar("Debug/Heading_Error_Deg", torch.rad2deg(heading_error.abs()).mean(), self.global_step)
            self._safe_add_scalar("Debug/Progress_Reward", progress_reward.mean(), self.global_step)
            self._safe_add_scalar("Debug/V_Forward", v_forward.mean(), self.global_step)
            self._safe_add_scalar("Debug/Alignment", alignment.mean(), self.global_step)
        

        # === 动作惩罚 ===
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_magnitude = torch.norm(self._actions, dim=1)
        action_rate_penalty = -torch.tanh(action_rate)
        action_mag_penalty = -torch.tanh(action_magnitude)

        # === 教师奖励 ===
        imitate_loss = torch.sum((self._actions - self._actions) ** 2, dim=1)
        imitate_reward = -imitate_loss
        # === 合并奖励 ===
        rewards = {
            "progress_reward": progress_reward * self.cfg.traj_track_scale * self.step_dt,
            "lateral_penalty": -lateral_error * self.cfg.lateral_error_scale * self.step_dt,  # 添加 config 项
            "direction_reward": direction_reward * self.cfg.direction_scale * self.step_dt,
            "goal_bias": bias * self.cfg.traj_done_bonus,
            "action_rate_penalty": action_rate_penalty * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_mag_penalty": action_mag_penalty * self.cfg.action_magnitude_scale * self.step_dt,
            "imitation_reward": imitate_reward * self.cfg.imitation_scale * self.step_dt
        }
        
        # === 应用奖励平滑处理 ===
        rewards = self._apply_reward_smoothing(rewards, target_switched)
        
        # 检查每个奖励组件
        for key, value in rewards.items():
            rewards[key] = self._check_numerical_stability(value, f'reward_{key}')
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # 使用统一的数值稳定性检查
        reward = self._check_numerical_stability(reward, 'reward')
        reward = torch.clamp(reward, -100.0, 100.0)
        
        # 额外的安全检查
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            print(f"严重警告：奖励中仍有NaN/Inf，步数：{self.global_step}")
            reward = torch.zeros_like(reward)
        # === 详细指标计算 ===
        # 计算距离指标
        current_pos = self._robot.data.root_state_w[:, :2]
        # 计算距离前检查
        pos_diff = current_pos - self.last_pos
        pos_diff = self._check_numerical_stability(pos_diff, 'pos_diff_for_distance')
        distance_traveled = torch.norm(pos_diff, dim=1)
        distance_traveled = self._check_numerical_stability(distance_traveled, 'distance_traveled')
        self._episode_metrics["total_distance"] += distance_traveled
        
        # 计算速度指标
        # 计算速度前检查
        vel_data = self._robot.data.root_state_w[:, 7:9]
        vel_data = self._check_numerical_stability(vel_data, 'vel_data_before_norm')
        current_vel = torch.norm(vel_data, dim=1)
        current_vel = self._check_numerical_stability(current_vel, 'current_vel')
        # 防止除以零：如果global_step为0，直接使用current_vel
        divisor = torch.clamp(torch.tensor(self.global_step + 1, device=self.device, dtype=torch.float32), min=1.0)
        self._episode_metrics["avg_speed"] = (self._episode_metrics["avg_speed"] * self.global_step + current_vel) / divisor
        
        # 计算侧向误差
        lateral_error_abs = lateral_error.abs()
        self._episode_metrics["max_lateral_error"] = torch.max(self._episode_metrics["max_lateral_error"], lateral_error_abs)
        
        # 计算最终距离
        self._episode_metrics["final_distance"] = dist_to_target
        
        # 更新位置记录
        self.last_pos = current_pos.clone()
        
        # === 日志记录 ===
        # 记录奖励指标（使用安全方法防止NaN/Inf）
        for key, value in rewards.items():
            self._episode_sums[key] = self._episode_sums.get(key, torch.zeros_like(value)) + value
            self._safe_add_scalar(f"Reward/{key}", value.mean(), self.global_step)

        self._safe_add_scalar("Reward/Total", reward.mean(), self.global_step)
        
        # 记录环境指标
        self._safe_add_scalar("Environment/Distance_to_Target", dist_to_target.mean(), self.global_step)
        self._safe_add_scalar("Environment/Lateral_Error", lateral_error.abs().mean(), self.global_step)
        self._safe_add_scalar("Environment/Heading_Error", heading_error.abs().mean(), self.global_step)
        self._safe_add_scalar("Environment/Robot_Speed", current_vel.mean(), self.global_step)
        self._safe_add_scalar("Environment/Action_Magnitude", action_magnitude.mean(), self.global_step)
        self._safe_add_scalar("Environment/Action_Rate", action_rate.mean(), self.global_step)
        
        # 记录平滑相关指标
        self._safe_add_scalar("Smoothing/Target_Switch_Count", target_switched.sum(), self.global_step)
        self._safe_add_scalar("Smoothing/Current_WP_Index", self._current_wp_idx.float().mean(), self.global_step)
        
        # 记录数值稳定性指标
        for key, count in self.nan_inf_count.items():
            self._safe_add_scalar(f"Debug/{key}_nan_inf_count", count, self.global_step)
        
        # 记录课程学习指标
        if self.curriculum_enabled:
            self._safe_add_scalar("Curriculum/Current_Stage", self.curriculum_stage, self.global_step)
            self._safe_add_scalar("Curriculum/Episode_Count", self.episode_count, self.global_step)
            self._safe_add_scalar("Curriculum/Num_Waypoints", self.cfg.num_waypoints, self.global_step)
            self._safe_add_scalar("Curriculum/Num_Interp", self.cfg.num_interp, self.global_step)
            self._safe_add_scalar("Curriculum/Step_Size", self.cfg.step_size, self.global_step)
            self._safe_add_scalar("Curriculum/Episode_Length", self.cfg.episode_length_s, self.global_step)
        
        # 记录性能指标
        self._safe_add_scalar("Performance/Episode_Length", self.global_step, self.global_step)
        self._safe_add_scalar("Performance/Total_Distance", self._episode_metrics["total_distance"].mean(), self.global_step)
        self._safe_add_scalar("Performance/Average_Speed", self._episode_metrics["avg_speed"].mean(), self.global_step)
        self._safe_add_scalar("Performance/Max_Lateral_Error", self._episode_metrics["max_lateral_error"].mean(), self.global_step)
        
        # 性能统计
        step_time = time.time() - step_start_time
        self._performance_stats["step_time"] = step_time
        self._performance_stats["fps"] = 1.0 / step_time if step_time > 0 else 0.0
        
        # 记录性能指标到 TensorBoard
        self._safe_add_scalar("Performance/Step_Time", step_time, self.global_step)
        self._safe_add_scalar("Performance/FPS", self._performance_stats["fps"], self.global_step)
        
        # 每100步记录一次详细统计
        if self.global_step % 100 == 0:
            elapsed_time = time.time() - self.begin_time
            self._safe_add_scalar("Training/Elapsed_Time", elapsed_time, self.global_step)
            steps_per_second = self.global_step / elapsed_time if elapsed_time > 0 else 0.0
            self._safe_add_scalar("Training/Steps_Per_Second", steps_per_second, self.global_step)

        return reward
    # 7. 判断回合结束
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 7.1 判断是否达到最大回合步数
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        reached_target = torch.zeros_like(time_out, dtype=torch.bool)
        if self.dist_to_target is not None:
            reached_target = self.dist_to_target < 0.2
            self._current_wp_idx[reached_target] += 1
        finished_traj = self._current_wp_idx >= self._trajectories.shape[1]
        self.finished_mask = finished_traj
        self.count += sum(finished_traj)
        
        # 记录回合结束统计
        episode_dones = finished_traj | time_out
        if episode_dones.any():
            self._log_episode_statistics(episode_dones)
            
            # 更新课程学习统计
            if self.curriculum_enabled:
                for i, done in enumerate(episode_dones):
                    if done:
                        success = finished_traj[i].item()  # 是否成功完成轨迹
                        self._update_curriculum_stats(success)
        
        self.last_pos = self.positions
        return finished_traj, time_out

    def _log_episode_statistics(self, episode_dones: torch.Tensor):
        """记录回合结束时的统计信息"""
        if not episode_dones.any():
            return
            
        # 计算成功率
        success_rate = (self._current_wp_idx >= self._trajectories.shape[1]).float().mean().item()
        
        # 记录回合统计
        for i, done in enumerate(episode_dones):
            if done:
                # 记录每个环境的回合统计
                episode_reward = sum(self._episode_sums[key][i].item() for key in self._episode_sums.keys())
                episode_length = self.episode_length_buf[i].item()
                
                # 更新训练统计
                self._training_stats["total_episodes"] += 1
                self._training_stats["best_reward"] = max(self._training_stats["best_reward"], episode_reward)
                # 防止除以零：确保total_episodes > 0
                total_eps = max(self._training_stats["total_episodes"], 1)
                self._training_stats["avg_reward"] = (
                    (self._training_stats["avg_reward"] * (total_eps - 1) + episode_reward) 
                    / total_eps
                )
                
                # 记录到 TensorBoard（使用安全方法防止NaN/Inf）
                self._safe_add_scalar("Episode/Episode_Reward", episode_reward, self._training_stats["total_episodes"])
                self._safe_add_scalar("Episode/Episode_Length", episode_length, self._training_stats["total_episodes"])
                self._safe_add_scalar("Episode/Success_Rate", success_rate, self._training_stats["total_episodes"])
                self._safe_add_scalar("Episode/Total_Distance", self._episode_metrics["total_distance"][i], self._training_stats["total_episodes"])
                self._safe_add_scalar("Episode/Final_Distance", self._episode_metrics["final_distance"][i], self._training_stats["total_episodes"])
                self._safe_add_scalar("Episode/Max_Lateral_Error", self._episode_metrics["max_lateral_error"][i], self._training_stats["total_episodes"])
                
                # 记录各奖励分量
                for key in self._episode_sums.keys():
                    self._safe_add_scalar(f"Episode_Reward/{key}", self._episode_sums[key][i], self._training_stats["total_episodes"])
        
        # 记录全局统计
        self._safe_add_scalar("Training/Best_Reward", self._training_stats["best_reward"], self.global_step)
        self._safe_add_scalar("Training/Average_Reward", self._training_stats["avg_reward"], self.global_step)
        self._safe_add_scalar("Training/Total_Episodes", self._training_stats["total_episodes"], self.global_step)
        
        # 重置回合指标
        for key in self._episode_sums.keys():
            self._episode_sums[key][episode_dones] = 0.0
        for key in self._episode_metrics.keys():
            self._episode_metrics[key][episode_dones] = 0.0

    # 8. 环境重置
    def draw_spline(self, traj_points, color=(0.0, 1.0, 0.0), thickness=1.0):
        """
        使用 debug_draw 绘制 spline 曲线
        traj_points: torch.Tensor (N, 2) or (N, 3)
        """
        import carb
        if traj_points.shape[1] == 2:
            z = torch.full((traj_points.shape[0], 1), 0.05, device=traj_points.device)
            traj_points = torch.cat([traj_points, z], dim=1)

        # 转换为 List[carb.Float3]
        points = [carb.Float3(*p.cpu().tolist()) for p in traj_points]

        self.debug_draw.draw_lines_spline(points, carb.ColorRgba(*color, 1.0), thickness, False)

    def generate_random_walk_trajectory(self,start_pos, num_points=2, step_size=1.0, seed=42, num_interp=1):
        torch.manual_seed(seed)
        traj = [start_pos]
        for _ in range(num_points - 1):
            angle = torch.rand(1) * 2 * math.pi
            direction = torch.tensor([torch.cos(angle), torch.sin(angle)], device=self.device)
            new_point = traj[-1] + direction * step_size
            traj.append(new_point)
        points = torch.stack(traj).cpu().numpy()
        t = np.linspace(0, 1, len(points))
        cs_x = CubicSpline(t, points[:, 0])
        cs_y = CubicSpline(t, points[:, 1])
        
        t_new = np.linspace(0, 1, len(points) * num_interp)
        x_new = cs_x(t_new)
        y_new = cs_y(t_new)
        smoothed = np.stack([x_new, y_new], axis=1)
        
        # 检查生成的轨迹是否有NaN或Inf
        if np.any(np.isnan(smoothed)) or np.any(np.isinf(smoothed)):
            print(f"警告：轨迹生成中出现NaN/Inf，使用简单直线轨迹替代")
            # 生成简单的直线轨迹作为备用
            simple_traj = np.linspace(start_pos.cpu().numpy(), start_pos.cpu().numpy() + [step_size, 0], num_points * num_interp)
            smoothed = simple_traj

        traj_tensor = torch.tensor(smoothed, dtype=torch.float32, device=self.device)
        return self._check_numerical_stability(traj_tensor, 'trajectory')



    def _reset_idx(self, env_ids: torch.Tensor | None):
        # 8.1 获取需要重置的环境 ID
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        if self.finished_mask is None:
            self.finished_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=env_ids.device)
        # 8.2 重置机器人状态
        self._robot.reset(env_ids)
        # 8.3 重置基础环境状态
        super()._reset_idx(env_ids)
        
        # 8.4 重置奖励平滑状态
        if env_ids is not None:
            self._prev_wp_idx[env_ids] = 0
            self._target_switch_detected[env_ids] = False
            # 重置历史奖励
            for key in self._prev_rewards:
                if self._prev_rewards[key] is not None:
                    self._prev_rewards[key][env_ids] = 0.0
        
        # 8.5 更新回合计数和课程学习阶段
        if env_ids is not None and len(env_ids) > 0:
            self.episode_count += len(env_ids)
            # 更新课程学习阶段
            self._update_curriculum_stage()

        # 8.4 重置动作缓冲
        default_root_state = self._robot.data.default_root_state[env_ids]
        self._prev_dist_to_target[env_ids] = 0.0
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        if self.debug_draw:
            self.debug_draw.clear_lines()
        # 使用课程学习的参数
        if self.curriculum_enabled:
            stage_config = self.curriculum_stages[self.curriculum_stage]
            num_points = stage_config['num_waypoints']
            num_interp = stage_config['num_interp']
            step_size = stage_config['step_size']
        else:
            num_points = self.cfg.num_waypoints
            num_interp = self.cfg.num_interp
            step_size = self.cfg.step_size
            
        for i, env_id in enumerate(env_ids):
            start = self._robot.data.root_state_w[env_id, :2]

            # 随机生成终点
            traj = self._trajectories[env_id]
            if self.finished_mask[env_id] or traj.abs().sum() == 0:
                traj = self.generate_random_walk_trajectory(start, num_points=num_points, num_interp=num_interp,
                                                         step_size=step_size, seed=random.randint(1, 100))
                self._trajectories[env_id] = traj
            self._current_wp_idx[env_id] = 0

            # 可选可视化
            if self.debug_draw:
                self.draw_spline(traj)
    
    def _apply_reward_smoothing(self, rewards: dict, target_switched: torch.Tensor) -> dict:
        """应用奖励平滑处理，解决目标点切换时的奖励突变问题"""
        smoothed_rewards = {}
        
        for key, reward in rewards.items():
            # 检查当前奖励的数值稳定性
            if torch.isnan(reward).any() or torch.isinf(reward).any():
                print(f"NaN/Inf found in {key} reward at step {self.global_step}, replacing with zeros")
                reward = torch.where(torch.isnan(reward) | torch.isinf(reward), torch.zeros_like(reward), reward)
            
            # 检查是否有历史奖励数据
            if key in self._prev_rewards and self._prev_rewards[key] is not None:
                # 检查历史奖励的数值稳定性
                if torch.isnan(self._prev_rewards[key]).any() or torch.isinf(self._prev_rewards[key]).any():
                    print(f"NaN/Inf found in previous {key} reward at step {self.global_step}, resetting")
                    self._prev_rewards[key] = torch.zeros_like(reward)
                
                # 根据是否发生目标点切换选择平滑因子
                if target_switched.any():
                    # 目标点切换时使用更大的平滑因子
                    smooth_factor = self.transition_smoothing_factor
                else:
                    # 正常情况使用标准平滑因子
                    smooth_factor = self.smoothing_factor
                
                # 应用指数移动平均平滑
                smoothed_reward = (1 - smooth_factor) * self._prev_rewards[key] + smooth_factor * reward
                smoothed_rewards[key] = smoothed_reward
            else:
                # 第一次计算，直接使用当前奖励
                smoothed_rewards[key] = reward
            
            # 更新历史奖励
            self._prev_rewards[key] = reward.clone()
        
        return smoothed_rewards
    
    def _check_numerical_stability(self, tensor, name, step=None):
        """检查张量的数值稳定性，并记录详细的溯源信息，并自动修复NaN/Inf"""
        # 检查是否为 torch.Tensor
        if not isinstance(tensor, torch.Tensor):
            return tensor
        
        # 立即检测NaN/Inf（不依赖debug_mode，确保总是检测）
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        # 保存原始tensor用于调试（在修复前）
        original_tensor = tensor.clone() if (has_nan or has_inf) else None
        
        # 如果检测到NaN/Inf，立即修复（替换为零）
        if has_nan or has_inf:
            # 立即修复，防止NaN传播
            tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), 
                               torch.zeros_like(tensor), tensor)
        
        # 记录和打印信息（使用原始tensor，在修复前收集信息）
        if (has_nan or has_inf) and hasattr(self, '_always_check_nan') and self._always_check_nan:
            if step is None:
                step = getattr(self, 'global_step', 0)
            
            # 初始化计数字典（如果不存在）
            if not hasattr(self, 'nan_inf_count'):
                self.nan_inf_count = {}
            if name not in self.nan_inf_count:
                self.nan_inf_count[name] = 0
                
            self.nan_inf_count[name] += 1
            
            # 获取调用栈信息（跳过当前函数和调用它的函数）
            stack = traceback.extract_stack()
            # 获取调用者的信息（跳过当前函数和调用它的函数）
            caller_info = stack[-3] if len(stack) >= 3 else stack[-2] if len(stack) >= 2 else None
            
            # 收集详细的调试信息（使用原始tensor，修复前的）
            debug_tensor = original_tensor if original_tensor is not None else tensor
            nan_count = torch.isnan(debug_tensor).sum().item() if has_nan else 0
            inf_count = torch.isinf(debug_tensor).sum().item() if has_inf else 0
            total_elements = debug_tensor.numel()
            
            # 获取有效值的统计信息（排除NaN/Inf）
            valid_tensor = debug_tensor[~(torch.isnan(debug_tensor) | torch.isinf(debug_tensor))]
            if len(valid_tensor) > 0:
                valid_min = valid_tensor.min().item()
                valid_max = valid_tensor.max().item()
                valid_mean = valid_tensor.mean().item()
                valid_std = valid_tensor.std().item()
            else:
                valid_min = valid_max = valid_mean = valid_std = float('nan')
            
            # 构建溯源信息
            trace_info = {
                'step': step,
                'variable_name': name,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'nan_count': nan_count,
                'inf_count': inf_count,
                'total_elements': total_elements,
                # 防止除以零：确保total_elements > 0
                'nan_ratio': float(nan_count) / float(total_elements) if total_elements > 0 else 0.0,
                'inf_ratio': float(inf_count) / float(total_elements) if total_elements > 0 else 0.0,
                'shape': list(debug_tensor.shape),
                'dtype': str(debug_tensor.dtype),
                'device': str(debug_tensor.device),
                'valid_min': valid_min,
                'valid_max': valid_max,
                'valid_mean': valid_mean,
                'valid_std': valid_std,
                'caller_file': caller_info.filename if caller_info else 'unknown',
                'caller_line': caller_info.lineno if caller_info else 0,
                'caller_function': caller_info.name if caller_info else 'unknown',
                'caller_code': caller_info.line if caller_info else 'unknown',
                'total_occurrences': self.nan_inf_count[name],
                'timestamp': time.time()
            }
            
            # 记录到溯源日志
            self.nan_trace_log.append(trace_info)
            # 只保留最近1000条记录，避免内存溢出
            if len(self.nan_trace_log) > 1000:
                self.nan_trace_log = self.nan_trace_log[-1000:]
            
            # 如果是第一次出现，记录到首次出现字典
            if name not in self.nan_first_occurrence:
                self.nan_first_occurrence[name] = trace_info.copy()
                self.nan_first_occurrence[name]['is_first'] = True
            
            # 打印详细的溯源信息
            print(f"\n{'='*80}")
            print(f"⚠️  NaN/Inf 检测到! 变量: {name}")
            print(f"{'='*80}")
            print(f"步骤: {step} | 总计出现次数: {self.nan_inf_count[name]}")
            # 防止除以零：计算百分比前检查
            nan_pct = (nan_count / total_elements * 100) if total_elements > 0 else 0.0
            inf_pct = (inf_count / total_elements * 100) if total_elements > 0 else 0.0
            print(f"NaN数量: {nan_count}/{total_elements} ({nan_pct:.2f}%)")
            print(f"Inf数量: {inf_count}/{total_elements} ({inf_pct:.2f}%)")
            print(f"形状: {debug_tensor.shape} | 类型: {debug_tensor.dtype} | 设备: {debug_tensor.device}")
            
            if len(valid_tensor) > 0:
                print(f"有效值范围: [{valid_min:.6f}, {valid_max:.6f}]")
                print(f"有效值均值: {valid_mean:.6f} ± {valid_std:.6f}")
            else:
                print("⚠️  警告: 所有值都是NaN或Inf!")
            
            if caller_info:
                print(f"\n调用位置:")
                print(f"  文件: {caller_info.filename}")
                print(f"  行号: {caller_info.lineno}")
                print(f"  函数: {caller_info.name}")
                print(f"  代码: {caller_info.line}")
            
            # 如果是第一次出现，特别标注
            if name in self.nan_first_occurrence and self.nan_first_occurrence[name].get('is_first'):
                print(f"\n🔍 这是变量 '{name}' 第一次出现 NaN/Inf")
                self.nan_first_occurrence[name]['is_first'] = False
            
            print(f"{'='*80}\n")
            
            # 再次确保替换NaN和Inf值（双重保险）
            tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), 
                               torch.zeros_like(tensor), tensor)
        
        # 最终验证：确保返回的张量没有NaN/Inf
        if isinstance(tensor, torch.Tensor):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"⚠️  警告：{name} 在修复后仍有NaN/Inf，强制替换为零")
                tensor = torch.where(torch.isnan(tensor) | torch.isinf(tensor), 
                                   torch.zeros_like(tensor), tensor)
        
        return tensor
    
    def _safe_add_scalar(self, tag, scalar_value, global_step, default_value=0.0):
        """安全地记录标量值到TensorBoard，自动处理NaN/Inf"""
        # 检查是否为torch.Tensor
        if isinstance(scalar_value, torch.Tensor):
            if scalar_value.numel() == 1:
                scalar_value = scalar_value.item()
            else:
                # 如果是多元素张量，计算均值
                scalar_value = scalar_value.mean().item()
        
        # 检查是否为NaN或Inf
        is_invalid = (scalar_value is None or 
                     (isinstance(scalar_value, float) and 
                      (math.isnan(scalar_value) or math.isinf(scalar_value))))
        
        if is_invalid:
            # 如果值为NaN/Inf，使用默认值并记录警告
            if hasattr(self, 'nan_inf_count'):
                if 'tensorboard' not in self.nan_inf_count:
                    self.nan_inf_count['tensorboard'] = 0
                self.nan_inf_count['tensorboard'] += 1
            scalar_value = default_value
        
        # 记录到TensorBoard
        if hasattr(self, 'writer') and self.writer is not None:
            try:
                self.writer.add_scalar(tag, scalar_value, global_step)
            except Exception as e:
                # 如果记录失败，打印警告但不中断训练
                print(f"警告：无法记录 {tag} 到TensorBoard: {e}")
        
        return scalar_value
    
    def _init_curriculum_parameters(self):
        """初始化课程学习参数"""
        if not self.curriculum_enabled:
            return
        
        # 应用当前阶段的参数
        self._apply_curriculum_stage_config()
        
        # 记录课程学习信息
        print(f"\n{'='*80}")
        print(f"课程学习已启用")
        print(f"当前阶段: {self.curriculum_stage} - {self.curriculum_stages[self.curriculum_stage]['stage_name']}")
        print(f"成功率阈值: {self.curriculum_success_rate_threshold:.1%}")
        print(f"窗口大小: {self.curriculum_success_window_size} 回合")
        print(f"各阶段最小回合数: {self.curriculum_min_episodes_per_stage}")
        print(f"{'='*80}\n")
    
    def _update_curriculum_stage(self):
        """更新课程学习阶段"""
        if not self.curriculum_enabled:
            return
        
        # 检查是否需要切换阶段
        old_stage = self.curriculum_stage
        new_stage = self._determine_curriculum_stage()
        
        if new_stage != old_stage:
            self.curriculum_stage = new_stage
            self._apply_curriculum_stage_config()
            self._log_curriculum_stage_change(old_stage, new_stage)
    
    def _determine_curriculum_stage(self):
        """确定当前应该处于的课程阶段（基于成功率和最小回合数）"""
        # 计算当前阶段的窗口成功率
        window_success_rate = self._get_window_success_rate()
        
        # 获取当前阶段的最小回合数
        min_episodes = self.curriculum_min_episodes_per_stage[self.curriculum_stage]
        
        # 判断是否可以切换到下一阶段
        can_progress = (
            window_success_rate >= self.curriculum_success_rate_threshold and
            self.curriculum_stats[f'stage_{self.curriculum_stage}_steps'] >= min_episodes
        )
        
        # 如果当前阶段已达标，尝试切换到下一阶段
        if can_progress and self.curriculum_stage < 2:
            return self.curriculum_stage + 1
        
        # 否则保持当前阶段
        return self.curriculum_stage
    
    def _get_window_success_rate(self):
        """计算最近窗口期内的成功率"""
        if len(self.success_history) < 10:  # 至少需要10个样本
            return 0.0
        
        # 取最近窗口期的成功/失败记录
        recent_history = self.success_history[-self.curriculum_success_window_size:]
        if len(recent_history) == 0:
            return 0.0
        
        # 计算成功率（确保数值稳定性）
        success_count = sum(recent_history)
        if success_count < 0 or len(recent_history) <= 0:
            return 0.0
        
        # 防止除以零：如果历史记录为空，返回0
        history_len = len(recent_history)
        if history_len == 0:
            success_rate = 0.0
        else:
            success_rate = float(success_count) / float(history_len)
        
        # 确保返回的值在有效范围内，并且不是NaN或Inf
        if math.isnan(success_rate) or math.isinf(success_rate):
            return 0.0
        
        # 限制在 [0, 1] 范围内
        success_rate = max(0.0, min(1.0, success_rate))
        return success_rate
    
    def _apply_curriculum_stage_config(self):
        """应用当前课程阶段的配置"""
        if not self.curriculum_enabled:
            return
        
        stage_config = self.curriculum_stages[self.curriculum_stage]
        
        # 更新环境参数
        self.cfg.num_waypoints = stage_config['num_waypoints']
        self.cfg.num_interp = stage_config['num_interp']
        self.cfg.step_size = stage_config['step_size']
        self.cfg.episode_length_s = stage_config['episode_length_s']
        
        # 更新奖励权重
        self.cfg.traj_track_scale = stage_config['traj_track_scale']
        self.cfg.lateral_error_scale = stage_config['lateral_error_scale']
        self.cfg.direction_scale = stage_config['direction_scale']
        
        # 重新初始化轨迹张量以适应新的参数
        new_traj_size = self.cfg.num_waypoints * self.cfg.num_interp
        if self._trajectories.shape[1] != new_traj_size:
            print(f"重新初始化轨迹张量: {self._trajectories.shape[1]} -> {new_traj_size}")
            self._trajectories = torch.zeros(self.num_envs, new_traj_size, 2, device=self.device)
            # 重置所有环境的轨迹
            self._reset_trajectories_all()
        
        # 注意：max_episode_length 是只读属性，通过修改 cfg.episode_length_s 来间接更新
        # self.max_episode_length 会在父类中自动计算
    
    def _reset_trajectories_all(self):
        """重置所有环境的轨迹"""
        for env_id in range(self.num_envs):
            start = self._robot.data.root_state_w[env_id, :2]
            traj = self.generate_random_walk_trajectory(
                start, 
                num_points=self.cfg.num_waypoints, 
                num_interp=self.cfg.num_interp,
                step_size=self.cfg.step_size, 
                seed=random.randint(1, 100)
            )
            self._trajectories[env_id] = traj
            self._current_wp_idx[env_id] = 0
    
    def _log_curriculum_stage_change(self, old_stage, new_stage):
        """记录课程阶段切换"""
        old_name = self.curriculum_stages[old_stage]['stage_name']
        new_name = self.curriculum_stages[new_stage]['stage_name']
        
        # 获取切换时的成功率信息
        window_success_rate = self._get_window_success_rate()
        old_stage_steps = self.curriculum_stats[f'stage_{old_stage}_steps']
        
        print(f"课程学习阶段切换: {old_stage}({old_name}) -> {new_stage}({new_name}) at episode {self.episode_count}")
        print(f"  成功率: {window_success_rate:.2%}, {old_stage}阶段回合数: {old_stage_steps}")
        
        # 记录到TensorBoard
        if hasattr(self, 'writer'):
            self._safe_add_scalar("Curriculum/Stage", new_stage, self.global_step)
            self._safe_add_scalar("Curriculum/Stage_Change", 1.0, self.global_step)
    
    def _update_curriculum_stats(self, episode_success):
        """更新课程学习统计"""
        if not self.curriculum_enabled:
            return
        
        # 更新当前阶段的步数
        stage_key = f'stage_{self.curriculum_stage}_steps'
        self.curriculum_stats[stage_key] += 1
        
        # 添加成功率历史记录
        self.success_history.append(1.0 if episode_success else 0.0)
        # 保持历史记录在合理范围内（最多保留5000个样本）
        if len(self.success_history) > 5000:
            self.success_history = self.success_history[-5000:]
        
        # 更新成功率（使用指数移动平均）
        success_rate_key = f'stage_{self.curriculum_stage}_success_rate'
        alpha = 0.01  # 平滑因子
        current_success_rate = self.curriculum_stats[success_rate_key]
        new_success_rate = (1 - alpha) * current_success_rate + alpha * (1.0 if episode_success else 0.0)
        self.curriculum_stats[success_rate_key] = new_success_rate
        
        # 计算窗口成功率
        window_success_rate = self._get_window_success_rate()
        
        # 记录到TensorBoard（使用安全方法防止NaN/Inf）
        if hasattr(self, 'writer'):
            self._safe_add_scalar(f"Curriculum/Stage_{self.curriculum_stage}_Success_Rate", 
                                 new_success_rate, self.global_step)
            self._safe_add_scalar(f"Curriculum/Stage_{self.curriculum_stage}_Steps", 
                                 self.curriculum_stats[stage_key], self.global_step)
            self._safe_add_scalar(f"Curriculum/Stage_{self.curriculum_stage}_Window_Success_Rate", 
                                 window_success_rate, self.global_step)
            self._safe_add_scalar("Curriculum/Overall_Window_Success_Rate", 
                                 window_success_rate, self.global_step)
            can_progress_val = (1.0 if (window_success_rate >= self.curriculum_success_rate_threshold 
                                       and self.curriculum_stats[stage_key] >= self.curriculum_min_episodes_per_stage[self.curriculum_stage]) else 0.0)
            self._safe_add_scalar("Curriculum/Can_Progress", can_progress_val, self.global_step)
    
    def get_curriculum_info(self):
        """获取课程学习信息"""
        if not self.curriculum_enabled:
            return {"enabled": False}
        
        current_stage = self.curriculum_stages[self.curriculum_stage]
        return {
            "enabled": True,
            "current_stage": self.curriculum_stage,
            "stage_name": current_stage['stage_name'],
            "num_waypoints": current_stage['num_waypoints'],
            "num_interp": current_stage['num_interp'],
            "step_size": current_stage['step_size'],
            "episode_length_s": current_stage['episode_length_s'],
            "stats": self.curriculum_stats.copy()
        }
    
    def get_nan_trace_summary(self, variable_name=None, recent_only=True, max_items=50):
        """获取NaN/Inf溯源摘要
        
        Args:
            variable_name: 如果指定，只返回该变量的溯源信息
            recent_only: 是否只返回最近的记录
            max_items: 最大返回记录数
        
        Returns:
            包含溯源信息的字典
        """
        if not hasattr(self, 'nan_trace_log'):
            return {"error": "溯源系统未初始化"}
        
        # 筛选记录
        filtered_log = self.nan_trace_log
        if variable_name:
            filtered_log = [log for log in self.nan_trace_log if log['variable_name'] == variable_name]
        
        # 如果只返回最近的
        if recent_only:
            filtered_log = filtered_log[-max_items:]
        
        # 统计摘要
        summary = {
            "total_events": len(self.nan_trace_log),
            "filtered_events": len(filtered_log),
            "variables_with_nan": list(set(log['variable_name'] for log in self.nan_trace_log)),
            "first_occurrences": {},
            "recent_traces": filtered_log[-max_items:] if filtered_log else []
        }
        
        # 添加每个变量的首次出现信息
        for var_name, first_info in self.nan_first_occurrence.items():
            summary["first_occurrences"][var_name] = {
                "first_step": first_info.get('step', 0),
                "first_caller": first_info.get('caller_function', 'unknown'),
                "first_file": first_info.get('caller_file', 'unknown'),
                "first_line": first_info.get('caller_line', 0),
                "total_count": self.nan_inf_count.get(var_name, 0)
            }
        
        return summary
    
    def print_nan_trace_summary(self, variable_name=None):
        """打印NaN/Inf溯源摘要到控制台"""
        summary = self.get_nan_trace_summary(variable_name=variable_name)
        
        print(f"\n{'='*80}")
        print(f"NaN/Inf 溯源摘要")
        print(f"{'='*80}")
        print(f"总事件数: {summary['total_events']}")
        print(f"涉及变量数: {len(summary['variables_with_nan'])}")
        print(f"\n涉及的变量: {', '.join(summary['variables_with_nan'])}")
        
        print(f"\n{'='*80}")
        print(f"首次出现位置:")
        print(f"{'='*80}")
        for var_name, info in summary['first_occurrences'].items():
            print(f"\n变量: {var_name}")
            print(f"  首次出现步数: {info['first_step']}")
            print(f"  总计出现次数: {info['total_count']}")
            print(f"  首次调用位置:")
            print(f"    文件: {info['first_file']}")
            print(f"    行号: {info['first_line']}")
            print(f"    函数: {info['first_caller']}")
        
        if summary['recent_traces']:
            print(f"\n{'='*80}")
            print(f"最近 {len(summary['recent_traces'])} 次事件:")
            print(f"{'='*80}")
            for i, trace in enumerate(summary['recent_traces'][-10:], 1):  # 只显示最近10次
                print(f"\n事件 #{i}:")
                print(f"  变量: {trace['variable_name']}")
                print(f"  步数: {trace['step']}")
                print(f"  NaN: {trace['nan_count']}/{trace['total_elements']} ({trace['nan_ratio']*100:.2f}%)")
                print(f"  Inf: {trace['inf_count']}/{trace['total_elements']} ({trace['inf_ratio']*100:.2f}%)")
                print(f"  位置: {trace['caller_file']}:{trace['caller_line']} in {trace['caller_function']}()")
        
        print(f"{'='*80}\n")
    
    def export_nan_trace_to_file(self, filepath, variable_name=None):
        """导出NaN/Inf溯源信息到JSON文件"""
        import json
        
        summary = self.get_nan_trace_summary(variable_name=variable_name, recent_only=False)
        
        # 确保文件路径存在
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"NaN/Inf 溯源信息已导出到: {filepath}")
        return filepath