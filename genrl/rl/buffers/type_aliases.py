from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from genrl.utils.common.type_aliases import ndArray


@dataclass(frozen=True)
class GenRLEpisodeData:
    """Contains the datasets data for a single episode.

    This is the object returned by :class:`minari.MinariDataset.sample_episodes`.
    """

    id: int
    seed: Optional[int]
    total_timesteps: int
    observations: ndArray
    actions: ndArray
    rewards: ndArray
    terminations: ndArray
    truncations: ndArray


@dataclass(frozen=True)
class GenRLBufferSample:
    """
    Contains the data for subtrajectories

    b: batch size
    l: subsequence length
    d: dimension
    """

    # Spec
    episode_ids: np.array  # The episode number where the subtrajectory sampled
    timestep_ranges: np.array
    subseq_len: np.array

    # RL-Common
    observations: ndArray  # [b, l, d]
    actions: ndArray  # [b, l, d]
    next_observations: ndArray  # [b, l, d]
    rewards: ndArray  # [b, l]
    terminations: ndArray  # [b, l]
    truncations: ndArray    # [b, l]
    masks: ndArray  # [b, l]  # Whether each component is zero padded or not
