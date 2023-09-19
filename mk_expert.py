from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

import gymnasium as gym


if __name__ == "__main__":
    # env = gym.make("Walker2d-v3")
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=50000,
    #     save_path="/home/jsw7460/sb3_saved_dataset",
    #     name_prefix="Walker2d-v3",
    #     save_replay_buffer=True,
    #     save_vecnormalize=True,
    # )
    # model = SAC("MlpPolicy", env, verbose=1, buffer_size=1_000_000)
    # model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)

    env = gym.make("Hopper-v3")
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="/home/jsw7460/sb3_saved_dataset",
        name_prefix="Hopper-v3",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    model = SAC("MlpPolicy", env, verbose=1, buffer_size=1_000_000)
    model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)