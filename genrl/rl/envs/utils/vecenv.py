from typing import Dict, Tuple
from typing import List

import numpy as np

from genrl.rl.envs.utils.history import GenRLHistoryEnv
from genrl.utils.common.type_aliases import GenRLEnvOutput

OBS_IDX = 0
REW_IDX = 1
TERMINATION_IDX = 2
TRUNCATION_IDX = 3
INFO_IDX = 4


class GenRLVecEnv:

    def __init__(self, envs: Tuple[GenRLHistoryEnv, ...], seed: int):
        self.envs = envs
        self.num_envs = len(envs)
        self.seed = seed

    def reset(self, *args, **kwargs) -> GenRLEnvOutput:
        results = [env.reset(*args, **kwargs, seed=self.seed) for env in self.envs]
        return GenRLEnvOutput.batch_stack([result for result in results])

    def step(self, actions: np.ndarray) -> Tuple[GenRLEnvOutput, np.ndarray, np.ndarray, List[Dict]]:
        assert actions.ndim == 3, "Action should have dimension 3 for VecEnv; [ith_env, subseq_len, dimension]"

        results = [env.step(actions[idx, -1]) for idx, env in enumerate(self.envs)]

        rewards = np.stack([result[REW_IDX] for result in results])
        print("Rewards", rewards)
        terminations = np.stack([result[TERMINATION_IDX] or result[TRUNCATION_IDX] for result in results])
        infos = [result[INFO_IDX] for result in results]

        return GenRLEnvOutput.batch_stack([result[0] for result in results]), rewards, terminations, infos

    def render_array(self):
        images = [env.render() for env in self.envs]
        return images

    def __iter__(self):
        return iter(self.envs)