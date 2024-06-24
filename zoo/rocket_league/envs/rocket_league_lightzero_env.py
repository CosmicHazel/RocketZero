import copy
from typing import List, Dict

import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

from wrappers.rocket_league_wrappers import wrap_rocket_league


@ENV_REGISTRY.register("rocket_league_lightzero")
class RocketLeagueEnvLightZero(BaseEnv):
    """
    A custom environment wrapper for Rocket League, designed to work with LightZero.
    This environment handles the alternating turns between blue and orange teams.
    """

    # Default configuration
    config = dict(
        collector_env_num=8,
        evaluator_env_num=3,
        n_evaluator_episode=3,
        env_id="RocketLeague-v0",
        obs_shape=372,
        collect_max_episode_steps=int(1.08e5),
        eval_max_episode_steps=int(1.08e5),
        render_mode_human=False,
        save_replay=False,
        replay_path=None,
        episode_life=True,
        clip_rewards=False,
        manager=dict(shared_memory=False),
        stop_value=int(1e6),
    )

    @classmethod
    def default_config(cls):
        """Returns the default configuration as an EasyDict."""
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + "Dict"
        return cfg

    def __init__(self, cfg: EasyDict):
        """Initialize the Rocket League environment."""
        self.cfg = cfg
        self._init_flag = False
        self.agents = ["blue", "orange"]
        self.num_agents = len(self.agents)
        self.possible_agents = self.agents[:]
        self.has_reset = False
        self._cumulative_rewards_next = None
        self._done_next = None
        self._info_next = None
        self.current_player = 0
        self.stored_action_blue = None
        self.stored_action_orange = None

    def reset(self):
        """Reset the environment and return the initial observation."""
        print("Resetting environment")
        if not self._init_flag:
            print("Initializing environment for the first time")
            self._env = wrap_rocket_league(self.cfg)
            self._observation_space = self._env.observation_space()[1]
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0],
                high=self._env.reward_range[1],
                shape=(1,),
                dtype=np.float32,
            )
            self._init_flag = True

        # Handle seeding
        if (
            hasattr(self, "_seed")
            and hasattr(self, "_dynamic_seed")
            and self._dynamic_seed
        ):
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
        elif hasattr(self, "_seed"):
            self._env.seed(self._seed)

        # Reset internal state
        self._cumulative_rewards_next = [0, 0]
        self._done_next = False
        self._info_next = {}

        # Create action mask
        action_mask = (
            np.ones(self._action_space.n, "int8")
            if self._action_space.__class__.__name__ == "Discrete"
            else None
        )

        self.obs = self._env.reset()
        self.stored_action_blue = None
        self.stored_action_orange = None
        self.current_player = 0  # Start with blue team

        # Prepare observation dict for LightZero
        lightzero_obs_dict = {
            "observation": self.obs[1],
            "action_mask": action_mask,
            "to_play": self.current_player,
        }

        return lightzero_obs_dict

    def step(self, action):
        """
        Perform a step in the environment.
        This method handles the alternating turns between blue and orange teams.
        """
        if self.current_player == 0:
            # Blue team's turn
            self.stored_action_blue = action
            self.current_player = 1  # Switch to orange team

            # Prepare observation for orange team without stepping the environment
            action_mask = (
                np.ones(self._action_space.n, "int8")
                if isinstance(self._action_space, gym.spaces.Discrete)
                else None
            )
            lightzero_obs_dict = {
                "observation": self.obs[0],
                "action_mask": action_mask,
                "to_play": self.current_player,
            }

            return BaseEnvTimestep(
                lightzero_obs_dict,
                self._cumulative_rewards_next[0],
                self._done_next,
                self._info_next,
            )
        else:
            # Orange team's turn
            self.stored_action_orange = action

            # Prepare observation for next blue team turn
            action_mask = np.ones(self._action_space.n, "int8")
            lightzero_obs_dict = {
                "observation": self.obs[1],
                "action_mask": action_mask,
                "to_play": 0,  # Next turn is blue team
            }

            # TODO: Verify that rewards are being processed correctly

            # Switch back to blue team for next step
            self.current_player = 0

            # Step the environment with both teams' actions
            actions = np.concatenate(
                (self.stored_action_blue, self.stored_action_orange)
            )
            obs_next, rewards_next, done_next, truncated_next, info = self._env.step(
                actions
            )

            # Prepare return values
            done = self._done_next
            info = self._info_next.copy()
            cumulative_rewards = self._cumulative_rewards_next.copy()

            # Update internal state for next step
            self._done_next = done_next or truncated_next
            self._cumulative_rewards_next = [
                x + y for x, y in zip(self._cumulative_rewards_next, rewards_next)
            ]
            if done_next:
                info["eval_episode_return"] = self._cumulative_rewards_next[0]
            self._info_next = info
            self.obs = obs_next

            return BaseEnvTimestep(
                lightzero_obs_dict, cumulative_rewards[1], done, info
            )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return self._reward_space

    def close(self):
        """Close the environment and reset the initialization flag."""
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True):
        """Set the seed for this environment."""
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def __repr__(self):
        return f"LightZero Rocket League Env({self.cfg.env_id})"
