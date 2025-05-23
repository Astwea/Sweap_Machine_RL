from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
MYDOG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/astwea/MyDogTask/Mydog/source/Mydog/Mydog/assets/base_bot/base_bot.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_pos={
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
            "caster_wheel_joint": 0.0,
        },
    ),
    actuators={
    "wheels": ImplicitActuatorCfg(
        joint_names_expr=["left_wheel_joint", "right_wheel_joint", "caster_wheel_joint"],
        effort_limit_sim=6000.0,  # N·m
        velocity_limit_sim=10.0,  # rad/s
        stiffness=100.0,
        damping=1.0,
    ),
    },
)
SAODI_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"/home/astwea/MyDogTask/Mydog/source/Mydog/Mydog/assets/create_3.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)
