from typing import Any, Callable, Dict, Optional

import numpy as np

from genrl.rl.envs.utils.vecenv import GenRLVecEnv
from genrl.utils import interfaces
from genrl.utils.common.type_aliases import GenRLPolicyInput, GenRLEnvEvalResult


def evaluate_policy(
    model: "interfaces.PolicyPredictor",
    env: GenRLVecEnv,
    n_eval_episodes: int = 1,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    warn: bool = True,
) -> GenRLEnvEvalResult:

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    vis_observations = []

    episode_counts = np.zeros(n_envs, dtype="int")
    episode_count_targets = np.array([n_eval_episodes for _ in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()

    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    while (episode_counts < episode_count_targets).any():
        model_prediction = model.predict(GenRLPolicyInput.from_env_output(observations))
        action = model_prediction.pred_action
        new_observations, rewards, dones, infos = env.step(action)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            img = env.render_array()
            img_stack = np.stack(img, axis=0)  # [n_envs, x, y, 3]
            vis_observations.append(img_stack)

    if render:
        vis_observations = np.stack(vis_observations, axis=1)  # [n_envs, ep_len, x, y, 3]
        vis_observations = [vis_observations[i] for i in range(n_envs)]
    else:
        vis_observations = None

    eval_result = GenRLEnvEvalResult(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        vis_observations=vis_observations
    )

    return eval_result
