from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union, Any, List, Iterable, Tuple, Type

import numpy as np
from minari import MinariDataset
from minari.dataset.minari_storage import PathLike, MinariStorage

from genrl.rl.buffers.type_aliases import GenRLBufferSample, GenRLEpisodeData


@dataclass(frozen=True)
class SubTrajectoryData:
    """
        Contains the datasets data for subtrajectories.
    """
    total_timesteps: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    episode_ids: int
    next_observations: np.ndarray
    timestep_ranges: np.ndarray


class GenRLDataset(MinariDataset):
    def __init__(
        self,
        data: Union[MinariStorage, PathLike],
        seed: int,
        episode_indices: Optional[np.ndarray] = None,
        preprocess_obs: Optional[Callable[[Any], Any]] = lambda x: x,
    ):
        self.seed = seed
        super(GenRLDataset, self).__init__(data=data, episode_indices=episode_indices)

        self._generator = np.random.default_rng(seed)
        self.preprocess_obs = preprocess_obs

        self.episodes = None  # type: Union[None, List[GenRLEpisodeData, ...]]
        self._use_memory_cache = False

    def filter_episodes(
        self, condition: Callable[[GenRLEpisodeData], bool]
    ) -> GenRLDataset:
        """Filter the dataset episodes with a condition.

        The condition must be a callable which takes an `GenRLEpisodeData` instance and retutrns a bool.
        The callable must return a `bool` True if the condition is met and False otherwise.
        i.e filtering for episodes that terminate:

        ```
        dataset.filter(condition=lambda x: x['terminations'][-1] )
        ```

        Args:
            condition (Callable[[GenRLEpisodeData], bool]): callable that accepts any type(For our current backend, an h5py episode group) and returns True if certain condition is met.
        """

        def dict_to_episode_data_condition(episode: dict) -> bool:
            return condition(GenRLEpisodeData(**episode))

        mask = self._data.apply(
            dict_to_episode_data_condition, episode_indices=self._episode_indices
        )
        assert self._episode_indices is not None
        return GenRLDataset(
            self._data,
            seed=self.seed,
            episode_indices=self._episode_indices[mask],
            preprocess_obs=self.preprocess_obs
        )

    @classmethod
    def split_dataset(cls: Type[GenRLDataset], buffer: GenRLDataset, sizes: List[int]) -> Tuple[GenRLDataset, ...]:
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
            slice_dataset = cls(
                data=buffer.spec.data_path,
                seed=buffer.seed,
                episode_indices=indices[start_idx:end_idx],
                preprocess_obs=buffer.preprocess_obs)
            out_datasets.append(slice_dataset)
            start_idx = end_idx

        return tuple(out_datasets)

    def sample_episodes(self, n_episodes: int) -> Iterable[GenRLEpisodeData]:
        """Sample n number of episodes from the dataset.

        Args:
            n_episodes (Optional[int], optional): number of episodes to sample.
        """
        if self._use_memory_cache:
            return self._generator.choice(self.episodes, size=n_episodes, replace=True)

        else:
            indices = self._generator.choice(self.episode_indices, size=n_episodes, replace=True)
            episodes = self._data.get_episodes(indices)
            return list(map(lambda data: GenRLEpisodeData(**data), episodes))

    def cache_data(self):
        """Reading the hdf5 every time a trajectory is sampled is not good for speed and the lifespan of the disk.
        When this method is called, it stores the episode data inside the class in the form of a list.
        This increases memory usage but provides a faster sampling speed.
        """
        self.episodes = self.sample_episodes(n_episodes=len(self))
        self._use_memory_cache = True

    def sample_subtrajectories(
        self,
        n_episodes: int,
        subseq_len: int,
    ) -> GenRLBufferSample:

        episodes = self.sample_episodes(n_episodes)

        _thresholds = np.array([ep.total_timesteps - 1 for ep in episodes])
        start_idxs = self._generator.integers(0, _thresholds)

        ep_sample = next(iter(episodes))

        # Pre-allocate numpy arrays
        observations_batch = np.empty((n_episodes, subseq_len, ep_sample.observations.shape[-1]))
        next_observations_batch = np.empty_like(observations_batch)
        actions_batch = np.empty((n_episodes, subseq_len, ep_sample.actions.shape[-1]))
        rewards_batch = np.empty((n_episodes, subseq_len))
        terminations_batch = np.empty((n_episodes, subseq_len))
        truncations_batch = np.empty((n_episodes, subseq_len))
        masks_batch = np.empty((n_episodes, subseq_len))
        episode_ids_batch = np.empty((n_episodes,), dtype=int)
        timestep_ranges_batch = np.empty((n_episodes, subseq_len), dtype=int)

        for i, (ep, start_idx) in enumerate(zip(episodes, start_idxs)):
            end_idx = start_idx + subseq_len

            obs = ep.observations[start_idx: end_idx, ...]
            act = ep.actions[start_idx: end_idx, ...]
            next_obs = ep.observations[start_idx + 1: end_idx + 1, ...]
            rew = ep.rewards[start_idx: end_idx, ...]
            terminations = ep.terminations[start_idx: end_idx, ...]
            truncations = ep.truncations[start_idx: end_idx, ...]

            pad_size = subseq_len - len(act)
            obs_padding = np.zeros((subseq_len - len(obs), obs.shape[-1]))
            next_obs_padding = np.zeros((subseq_len - len(next_obs), obs.shape[-1]))
            act_padding = np.zeros((pad_size, act.shape[-1]))
            rew_padding = np.zeros((pad_size,))
            terminations_padding = np.zeros((pad_size,))
            truncations_padding = np.zeros((pad_size,))

            observations_batch[i] = np.concatenate((obs, obs_padding), axis=0)
            next_observations_batch[i] = np.concatenate((next_obs, next_obs_padding.copy()), axis=0)
            actions_batch[i] = np.concatenate((act, act_padding), axis=0)
            rewards_batch[i] = np.concatenate((rew, rew_padding), axis=0)
            terminations_batch[i] = np.concatenate((terminations, terminations_padding), axis=0)
            truncations_batch[i] = np.concatenate((truncations, truncations_padding), axis=0)
            masks_batch[i] = np.concatenate((np.ones((subseq_len - pad_size)), np.zeros((pad_size,))), axis=0)
            episode_ids_batch[i] = ep.id
            timestep_ranges_batch[i] = np.arange(start_idx, end_idx)

        data = {
            "subseq_len": subseq_len,
            "observations": observations_batch,
            "next_observations": next_observations_batch,
            "actions": actions_batch,
            "rewards": rewards_batch,
            "terminations": terminations_batch,
            "truncations": truncations_batch,
            "masks": masks_batch,
            "episode_ids": episode_ids_batch,
            "timestep_ranges": timestep_ranges_batch,
        }
        out = GenRLBufferSample(**data)
        return out
