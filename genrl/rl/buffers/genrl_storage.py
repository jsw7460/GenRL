import importlib.metadata
import os
from typing import Any, Callable, Iterable, List, Optional, Union

import h5py
import numpy as np
from minari.dataset.minari_storage import MinariStorage

# Use importlib due to circular import when: "from minari import __version__"
__version__ = importlib.metadata.version("minari")

PathLike = Union[str, bytes, os.PathLike]


class GenRLStorage(MinariStorage):
    """
        In GenRL, we also consider 'skill'
    """

    def __init__(self, data_path: PathLike, skill_based: bool = False):
        """Initialize properties of the Minari storage.

        Args:
            data_path (str): full path to the `main_data.hdf5` file of the dataset.
        """
        super(GenRLStorage, self).__init__(data_path)
        self.skill_based = False

    def apply(
        self,
        function: Callable[[dict], Any],
        episode_indices: Optional[Iterable] = None,
    ) -> List[Any]:
        """Apply a function to a slice of the data.

        Args:
            function (Callable): function to apply to episodes
            episode_indices (Optional[Iterable]): epsiodes id to consider

        Returns:
            outs (list): list of outputs returned by the function applied to episodes
        """
        if not self.skill_based:
            return super(GenRLStorage, self).apply(function, episode_indices)

        else:
            if episode_indices is None:
                episode_indices = range(self.total_episodes)
            out = []
            with h5py.File(self._data_path, "r") as file:
                for ep_idx in episode_indices:
                    ep_group = file[f"episode_{ep_idx}"]
                    assert isinstance(ep_group, h5py.Group)

                    ep_dict = {
                        "id": ep_group.attrs.get("id"),
                        "total_timesteps": ep_group.attrs.get("total_steps"),
                        "seed": ep_group.attrs.get("seed"),
                        "observations": self._decode_space(
                            ep_group["observations"], self.observation_space
                        ),
                        "actions": self._decode_space(
                            ep_group["actions"], self.action_space
                        ),
                        "sem_skills": ep_group["sem_skills"][()] if self.skill_based else np.zeros(1, ),
                        "sem_skills_done": ep_group["sem_skills_done"][()] if self.skill_based else np.zeros(1, ),
                        "rewards": ep_group["rewards"][()],
                        "terminations": ep_group["terminations"][()],
                        "truncations": ep_group["truncations"][()],
                    }
                    out.append(function(ep_dict))

            return out

    def get_episodes(self, episode_indices: Iterable[int]) -> List[dict]:
        """Get a list of episodes.

        Args:
            episode_indices (Iterable[int]): episodes id to return

        Returns:
            episodes (List[dict]): list of episodes data
        """
        out = []
        with h5py.File(self._data_path, "r") as file:
            for ep_idx in episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                out.append(
                    {
                        "id": ep_group.attrs.get("id"),
                        "total_timesteps": ep_group.attrs.get("total_steps"),
                        "seed": ep_group.attrs.get("seed"),
                        "observations": self._decode_space(
                            ep_group["observations"], self.observation_space
                        ),
                        "actions": self._decode_space(
                            ep_group["actions"], self.action_space
                        ),
                        "sem_skills": ep_group["sem_skills"][()] if self.skill_based else np.zeros(1, ),
                        "sem_skills_done": ep_group["sem_skills_done"][()] if self.skill_based else np.zeros(1, ),
                        "rewards": ep_group["rewards"][()],
                        "terminations": ep_group["terminations"][()],
                        "truncations": ep_group["truncations"][()],
                    }
                )

        return out
