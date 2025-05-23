# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Mydog-Marl-Direct-v0",
    entry_point=f"{__name__}.mydog_marl_env:MydogMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mydog_marl_env_cfg:MydogMarlEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_diff_drive_cfg.yaml",
    },
)
