from __future__ import annotations

from typing import Callable, Optional, Union, Any

import numpy as np
from jax.tree_util import tree_map

from genrl.rl.buffers.type_aliases import GenRLBufferSample
from minari import MinariDataset
from minari.dataset.minari_dataset import EpisodeData
from minari.dataset.minari_storage import MinariStorage, PathLike


class GenRLBuffer(MinariDataset):

    def __init__(
        self,
        data: Union[MinariStorage, PathLike],
        episode_indices: Optional[np.ndarray] = None,
    ):
        super(GenRLBuffer, self).__init__(data=data, episode_indices=episode_indices)

    def sample_subtrajectories(
        self,
        n_episodes: int,
        subseq_len: int,
        postprocess_fn: Optional[Callable[[dict], Any]],
        allow_replace: bool = False
    ) -> GenRLBufferSample:
        indices = self._generator.choice(self.episode_indices, size=n_episodes, replace=allow_replace)
        dict_data = self._data.apply(postprocess_fn, indices)
        episodes = [EpisodeData(**data) for data in dict_data]

        out = []

        for ep in episodes:
            start_idx = self._generator.choice(ep.total_timesteps - 1)
            end_idx = start_idx + subseq_len

            timesteps_range = np.arange(start_idx, end_idx)

            obs = ep.observations[start_idx: end_idx, ...]
            act = ep.actions[start_idx: end_idx, ...]

            next_obs = ep.observations[start_idx + 1: end_idx + 1, ...]
            rew = ep.rewards[start_idx: end_idx, ...]
            terminations = ep.terminations[start_idx: end_idx, ...]

            pad_size = subseq_len - len(act)
            obs_padding = np.zeros((subseq_len - len(obs), obs.shape[-1]))
            next_obs_padding = np.zeros((subseq_len - len(next_obs), obs.shape[-1]))
            act_padding = np.zeros((pad_size, act.shape[-1]))
            rew_padding = np.zeros((pad_size,))
            terminations_padding = np.zeros((pad_size,))

            observations = np.concatenate((obs, obs_padding), axis=0)
            actions = np.concatenate((act, act_padding), axis=0)
            next_observations = np.concatenate((next_obs, next_obs_padding.copy()), axis=0)
            rewards = np.concatenate((rew, rew_padding), axis=0)
            terminations = np.concatenate((terminations, terminations_padding), axis=0)
            masks = np.concatenate((np.ones((subseq_len - pad_size,)), np.zeros((pad_size,))), axis=0)

            subtraj_data = {
                "episode_id": ep.id,
                "timesteps_range": timesteps_range,
                "subseq_len": subseq_len,
                "observations": observations,
                "actions": actions,
                "next_observations": next_observations,
                "rewards": rewards,
                "terminations": terminations,
                "masks": masks
            }
            out.append(subtraj_data)

        out = tree_map(lambda *arr: np.stack(arr, axis=0), *out)
        out = GenRLBufferSample(**out)

        return out
