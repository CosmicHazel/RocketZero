import configparser
import os
from pydoc import render_doc
import time
from typing import Optional
import gym
from gym.envs.registration import register


import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import RepeatAction
from rlgym.rocket_league.done_conditions import (
    NoTouchTimeoutCondition,
    AnyCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.state_mutators import MutatorSequence
import torch

from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.rlviser import RLViserRenderer

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import (
    GoalCondition,
    AnyCondition,
    TimeoutCondition,
    NoTouchTimeoutCondition,
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.state_mutators import (
    MutatorSequence,
    FixedTeamSizeMutator,
    KickoffMutator,
)

config = configparser.ConfigParser()

config_file = "config.ini"
default_config_file = "config.default.ini"

if os.path.exists(config_file):
    config.read(config_file)
else:
    config.read(default_config_file)


BLUE_SIZE = int(config["Settings"]["BLUE_SIZE"])
ORANGE_SIZE = int(config["Settings"]["ORANGE_SIZE"])
ticks_per_step = int(config["Settings"]["ticks_per_step"])

# TODO change this to match your env because I was too lazy to pass this from the config
num_action_heads = 4
action_space_size = 324


# Define your custom Rocket League gym environment class here
class CustomRocketLeagueGymEnv(gym.Env):
    def __init__(self, render_mode="none"):
        super(CustomRocketLeagueGymEnv, self).__init__()
        self.num_action_heads = num_action_heads
        self.steps = 0
        self.action_space = gym.spaces.Discrete(action_space_size * num_action_heads)
        self.render_mode = render_mode
        self.env = RLGym(
            state_mutator=MutatorSequence(
                FixedTeamSizeMutator(blue_size=2, orange_size=2), KickoffMutator()
            ),
            obs_builder=DefaultObs(zero_padding=None),
            action_parser=RepeatAction(LookupTableAction(), repeats=8),
            reward_fn=CombinedReward((GoalReward(), 10.0), (TouchReward(), 0.1)),
            termination_cond=GoalCondition(),
            truncation_cond=AnyCondition(
                TimeoutCondition(timeout=300.0), NoTouchTimeoutCondition(timeout=30.0)
            ),
            transition_engine=RocketSimEngine(),
            renderer=RLViserRenderer(),
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        # Call the RLGym environment's reset method
        obs = torch.tensor(self.env.reset(), dtype=torch.float32)
        return obs

    def step(self, action):
        team_agents = {"blue": [], "orange": []}
        for agent in self.env.agents:
            if agent.startswith("blue"):
                team_agents["blue"].append(agent)
            elif agent.startswith("orange"):
                team_agents["orange"].append(agent)

        # Create a dictionary of actions for each agent
        actions = {}
        blue_team_action_index = 0
        orange_team_action_index = 4  # Start index for orange team

        # Distribute actions for the blue team
        for agent in team_agents["blue"]:
            actions[agent] = np.array([action[blue_team_action_index]])
            blue_team_action_index += 1

        # Distribute actions for the orange team
        for agent in team_agents["orange"]:
            actions[agent] = np.array([action[orange_team_action_index]])
            orange_team_action_index += 1

        # Call the superclass step method with the actions dictionary
        obs, rewards, terminated_dict, truncated_dict = self.env.step(actions)
        obs = torch.tensor(obs, dtype=torch.float32)
        self.render(self.render_mode)

        # Create the info dictionary for each agent
        info = {agent: {} for agent in self.env.agents}
        print(f"Steps: {self.steps}")
        self.steps += 1
        is_terminated = any(terminated_dict.values())
        is_truncated = any(truncated_dict.values())
        return obs, rewards, is_terminated, is_truncated, info

    def render(self, mode="human"):
        if mode == "human":
            self.env.render()

    def observation_space(self):
        return self.env.obs_builder.get_obs_space(None)


# Register the custom environment
register(
    id="CustomRocketLeagueEnv-v0",
    entry_point="custom_rocket_league_env:CustomRocketLeagueEnv",
)
