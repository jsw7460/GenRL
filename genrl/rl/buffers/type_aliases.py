from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch as th


@dataclass(frozen=True)
class GenRLBufferSample:
    """
    Contains the data for subtrajectories

    b: batch size
    l: subsequence length
    d: dimension
    """

    # Spec
    episode_id: np.array  # The episode number where the subtrajectory sampled
    timesteps_range: np.array
    subseq_len: np.array

    # RL-Common
    observations: np.ndarray | th.Tensor = np.empty(0, )  # [b, l, d]
    actions: np.ndarray | th.Tensor = np.empty(0, )  # [b, l, d]
    next_observations: np.ndarray | th.Tensor = np.empty(0, )  # [b, l, d]
    rewards: np.ndarray | th.Tensor = np.empty(0, )  # [b, l]
    terminations: np.ndarray | th.Tensor = np.empty(0, )  # [b, l]
    masks: np.ndarray | th.Tensor = np.empty(0, )  # [b, l]  # Whether each component is zero padded or not
