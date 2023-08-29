from __future__ import annotations

from typing import Callable, Optional, Union, Any, List

import numpy as np
from jax.tree_util import tree_map
from minari import MinariDataset
from minari.dataset.minari_dataset import EpisodeData
from minari.dataset.minari_storage import MinariStorage, PathLike

from genrl.rl.buffers.type_aliases import GenRLBufferSample


class GenRLDataset(MinariDataset):

    def __init__(
        self,
        data: Union[MinariStorage, PathLike],
        seed: int,
        episode_indices: Optional[np.ndarray] = None,
        postprocess_fn: Optional[Callable[[dict], Any]] = None,
    ):
        self.seed = seed
        super(GenRLDataset, self).__init__(data=data, episode_indices=episode_indices)
        self._generator = np.random.default_rng(seed)
        self.postprocess_fn = postprocess_fn

    def filter_episodes(
        self, condition: Callable[[EpisodeData], bool]
    ) -> GenRLDataset:
        """Filter the dataset episodes with a condition.

        The condition must be a callable which takes an `EpisodeData` instance and retutrns a bool.
        The callable must return a `bool` True if the condition is met and False otherwise.
        i.e filtering for episodes that terminate:

        ```
        dataset.filter(condition=lambda x: x['terminations'][-1] )
        ```

        Args:
            condition (Callable[[EpisodeData], bool]): callable that accepts any type(For our current backend, an h5py episode group) and returns True if certain condition is met.
        """

        def dict_to_episode_data_condition(episode: dict) -> bool:
            return condition(EpisodeData(**episode))

        mask = self._data.apply(
            dict_to_episode_data_condition, episode_indices=self._episode_indices
        )
        assert self._episode_indices is not None
        return GenRLDataset(
            self._data,
            seed=self.seed,
            episode_indices=self._episode_indices[mask],
            postprocess_fn=self.postprocess_fn
        )

    @classmethod
    def split_dataset(cls, buffer: GenRLDataset, sizes: List[int]):
        """Split a MinariDataset in multiple datasets.

            Args:
                buffer (GenRLDataset): the GenRLDataset to split
                sizes (List[int]): sizes of the resulting datasets

            Returns:
                datasets (List[MinariDataset]): resulting list of datasets
            """
        if sum(sizes) > buffer.total_episodes:
            raise ValueError(
                "Incompatible arguments: the sum of sizes exceeds ",
                f"the number of episodes in the dataset ({buffer.total_episodes})",
            )
        seed = buffer.seed
        generator = np.random.default_rng(seed=seed)
        indices = generator.permutation(buffer.episode_indices)
        out_datasets = []
        start_idx = 0
        for length in sizes:
            end_idx = start_idx + length
            slice_dataset = cls(buffer.spec.data_path, buffer.seed, indices[start_idx:end_idx], buffer.postprocess_fn)
            out_datasets.append(slice_dataset)
            start_idx = end_idx
        return out_datasets

    def sample_subtrajectories(
        self,
        n_episodes: int,
        subseq_len: int,
        allow_replace: bool = False
    ) -> GenRLBufferSample:
        indices = self._generator.choice(self.episode_indices, size=n_episodes, replace=allow_replace)
        dict_data = self._data.apply(self.postprocess_fn, indices)
        episodes = [EpisodeData(**data) for data in dict_data]

        _thresholds = np.array([ep.total_timesteps - 1 for ep in episodes])
        start_idxs = self._generator.integers(0, _thresholds)

        out = []

        for ep, start_idx in zip(episodes, start_idxs):
            # start_idx = self._generator.choice(ep.total_timesteps - 1)
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
