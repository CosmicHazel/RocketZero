from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
from ding.envs import ClipRewardWrapper
from easydict import EasyDict

from rocket_league_rlgym_gym_wrapper import CustomRocketLeagueGymEnv


def wrap_rocket_league(config: EasyDict) -> gym.Env:
    """
    Configure and wrap the Rocket League environment based on the provided configuration.

    This function creates a CustomRocketLeagueGymEnv instance and applies additional
    wrappers based on the configuration settings.

    Arguments:
        config (EasyDict): Dictionary containing configuration parameters for the environment.

    Returns:
        gym.Env: The wrapped Rocket League environment with the specified configurations.
    """

    # Create the base Rocket League environment
    env = CustomRocketLeagueGymEnv(
        render_mode="none" if not config.render_mode_human else "human"
    )

    # Apply ClipRewardWrapper if specified in the config
    if config.clip_rewards:
        env = ClipRewardWrapper(env)

    return env
