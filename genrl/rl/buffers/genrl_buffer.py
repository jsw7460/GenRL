from __future__ import annotations

from typing import Callable, Optional, Union, Any, List, Iterable, Tuple, Type

import numpy as np
from jax.tree_util import tree_map
from minari import MinariDataset
from minari.dataset.minari_storage import PathLike, MinariStorage

from vlg.rl.buffers.type_aliases import VLGBufferSample, VLGEpisodeData


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

        self.episodes = None  # type: Union[None, List[VLGEpisodeData, ...]]
        self.__use_memory_cache = False

    def filter_episodes(
        self, condition: Callable[[VLGEpisodeData], bool]
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
            return condition(VLGEpisodeData(**episode))

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

    def sample_episodes(self, n_episodes: int) -> Iterable[VLGEpisodeData]:
        """Sample n number of episodes from the dataset.

        Args:
            n_episodes (Optional[int], optional): number of episodes to sample.
        """
        if self.__use_memory_cache:
            return self._generator.choice(self.episodes, size=n_episodes, replace=True)

        else:
            indices = self._generator.choice(self.episode_indices, size=n_episodes, replace=True)
            episodes = self._data.get_episodes(indices)
            return list(map(lambda data: VLGEpisodeData(**data), episodes))

    def cache_data(self):
        """
            Reading the hdf5 every time a trajectory is sampled is not good for speed and the lifespan of the disk.
            When this method is called, it stores the episode data inside the class in the form of a list.
            This increases memory usage but provides a faster sampling speed.
        """
        self.episodes = self.sample_episodes(n_episodes=len(self))
        self.__use_memory_cache = True

    def sample_subtrajectories(
        self,
        n_episodes: int,
        subseq_len: int,
    ) -> VLGBufferSample:

        episodes = self.sample_episodes(n_episodes)

        _thresholds = np.array([ep.total_timesteps - 1 for ep in episodes])
        start_idxs = self._generator.integers(0, _thresholds)

        out = []

        for ep, start_idx in zip(episodes, start_idxs):
            end_idx = start_idx + subseq_len
            timesteps_range = np.arange(start_idx, end_idx)

            ep_obs = self.preprocess_obs(ep.observations)

            # obs = ep.observations["observation"]
            # obs = obs[start_idx: end_idx, ...]

            obs = ep_obs[start_idx: end_idx, ...]
            act = ep.actions[start_idx: end_idx, ...]

            # next_obs = obs[start_idx + 1: end_idx + 1, ...]

            next_obs = ep_obs[start_idx + 1: end_idx + 1, ...]
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
                "masks": masks,
            }
            out.append(subtraj_data)

        out = tree_map(lambda *arr: np.stack(arr, axis=0), *out)
        out = VLGBufferSample(**out)
        return out
