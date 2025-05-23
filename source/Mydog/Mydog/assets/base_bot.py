from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
MYDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/astwea/nav_ws/urdf/roomba.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
            static_friction=2.0,  # 静摩擦系数
            dynamic_friction=2.0,  # 动摩擦系数
            restitution=0.0,  # 弹性恢复系数
            mass=5.0,  # 设置质量
            inertia=(0.05, 0.05, 0.05),  # 确保惯性矩是正值
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
            "caster_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
    "wheels": ImplicitActuatorCfg(
        joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
        effort_limit=400.0,
        velocity_limit=100.0,
        stiffness=0.0,
        damping=10.0,
    ),
    },
)
