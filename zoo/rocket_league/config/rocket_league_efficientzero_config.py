import configparser
import os
from easydict import EasyDict

config = configparser.ConfigParser()

config_file = "../config/config.ini"
default_config_file = "../config/config.default.ini"

if os.path.exists(config_file):
    config.read(config_file)
else:
    config.read(default_config_file)

# Environment settings
env_id = "RocketLeague-v0"

# Action space configuration
# TODO: Adjust these values to match your specific environment
num_action_heads = int(config["Settings"]["num_action_heads"])
action_space_size = int(config["Settings"]["action_space_size"]) * num_action_heads

# Observation space configuration
# TODO: Adjust this value to match your specific environment
obs_shape = int(config["Settings"]["obs_shape"])

# Training parameters
collector_env_num = 2
n_episode = 2
evaluator_env_num = 2
num_simulations = 50
update_per_collect = 1000
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.0

eps_greedy_exploration_in_collect = False

# Main configuration dictionary
rocket_league_efficientzero_config = dict(
    exp_name=f"data_ez_ctree/{env_id}_efficientzero_ns{num_simulations}_upc{update_per_collect}_rr{reanalyze_ratio}_seed0",
    env=dict(
        battle_mode="self_play_mode",
        env_id=env_id,
        obs_shape=obs_shape,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(
            shared_memory=False,
        ),
    ),
    policy=dict(
        # Model configuration
        model=dict(
            observation_shape=obs_shape,
            frame_stack_num=2,
            action_space_size=action_space_size,
            downsample=False,
            lstm_hidden_size=256,
            latent_state_dim=256,
            discrete_action_encoding_type="one_hot",
            norm_type="BN",
        ),
        cuda=True,
        env_type="board_games",
        game_segment_length=1000,
        random_collect_episode_num=0,
        # Epsilon-greedy exploration settings
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            type="linear",
            start=1.0,
            end=0.05,
            decay=int(1e5),
        ),
        # Training settings
        use_augmentation=False,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        model_type="mlp",
        optim_type="Adam",
        lr_piecewise_constant_decay=True,
        learning_rate=0.2,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

# Convert the configuration to an EasyDict for easier access
rocket_league_efficientzero_config = EasyDict(rocket_league_efficientzero_config)
main_config = rocket_league_efficientzero_config

# Configuration for creating the environment and policy
rocket_league_efficientzero_create_config = dict(
    env=dict(
        type="rocket_league_lightzero",
        import_names=["rocket_league_lightzero_env"],
    ),
    env_manager=dict(type="subprocess"),
    policy=dict(
        type="efficientzero",
        import_names=["lzero.policy.efficientzero"],
    ),
)
rocket_league_efficientzero_create_config = EasyDict(
    rocket_league_efficientzero_create_config
)
create_config = rocket_league_efficientzero_create_config

if __name__ == "__main__":
    # Import and run the EfficientZero training function
    from lzero.entry import train_muzero

    train_muzero(
        [main_config, create_config],
        seed=0,
    )
