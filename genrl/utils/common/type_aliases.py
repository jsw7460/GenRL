from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union, Deque, List

import gymnasium as gym
import numpy as np
from jax import numpy as jnp

ndArray = Union[jnp.ndarray, np.ndarray]
GymEnv = gym.Env

PolicyOutput = Tuple[np.ndarray, Optional[Dict[str, Any]]]

nnOutput = Dict[str, jnp.ndarray]


@dataclass(frozen=True)
class GenRLEnvOutput:
    """
    Contains the data for subtrajectories

    b: batch size ( = n_envs in vectorized environments)
    l: subsequence length
    d: dimension
    """

    # Spec
    subseq_len: Union[int, List[int]]

    # RL History
    observations: ndArray  # [b, l, d]
    actions: ndArray  # [b, l, d]
    masks: ndArray  # [b, l]  # Whether each component is zero padded or not

    # Environment aware
    timesteps: ndArray  # [b, l] (History timesteps)
    rewards: ndArray  # [b, l]
    terminations: ndArray  # [b, l]
    truncations: ndArray  # [b, l]
    info: Union[List[Dict], Deque[Dict], List[List[Dict]]]

    # Maybe None
    sem_skills: Optional[ndArray] = None
    sem_skills_done: Optional[ndArray] = None

    @staticmethod
    def batch_stack(batch_output: List["GenRLEnvOutput"]) -> "GenRLEnvOutput":
        subseq_len = batch_output[0].subseq_len
        observations = np.stack([o.observations for o in batch_output], axis=0)
        actions = np.stack([o.actions for o in batch_output], axis=0)
        masks = np.stack([o.masks for o in batch_output], axis=0)

        timesteps = np.stack([o.timesteps for o in batch_output], axis=0)
        rewards = np.stack([o.rewards for o in batch_output], axis=0)
        terminations = np.stack([o.terminations for o in batch_output], axis=0)
        truncations = np.stack([o.truncations for o in batch_output], axis=0)
        info = [o.info for o in batch_output]

        if batch_output[0].sem_skills is not None:
            sem_skills = np.stack([o.sem_skills for o in batch_output])
            sem_skills_done = np.stack([o.sem_skills_done for o in batch_output])
        else:
            sem_skills = None
            sem_skills_done = None

        return GenRLEnvOutput(
            subseq_len=subseq_len,
            observations=observations,
            actions=actions,
            masks=masks,
            timesteps=timesteps,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            info=info,
            sem_skills=sem_skills,
            sem_skills_done=sem_skills_done
        )


@dataclass(frozen=True)
class GenRLPolicyInput:
    """
    Contains the data for subtrajectories

    b: batch size
    l: subsequence length
    d: dimension
    """

    # RL History
    observations: ndArray  # [b, l, d]
    actions: ndArray  # [b, l, d]
    masks: ndArray  # [b, l]  # Whether each component is zero padded or not

    rewards: ndArray  # [b, l]
    terminations: ndArray  # [b, l]
    timesteps: ndArray

    sem_skills: Optional[ndArray] = None
    sem_skills_done: Optional[ndArray] = None

    @staticmethod
    def from_env_output(x: GenRLEnvOutput) -> "GenRLPolicyInput":
        return GenRLPolicyInput(
            observations=x.observations,
            actions=x.actions,
            masks=x.masks,
            rewards=x.rewards,
            timesteps=x.timesteps,
            terminations=x.terminations,
            sem_skills=x.sem_skills,
            sem_skills_done=x.sem_skills_done
        )


@dataclass(frozen=True)
class GenRLPolicyOutput:
    pred_action: ndArray  # [b, l, d]
    info: Dict


@dataclass(frozen=True)
class GenRLEnvEvalResult:
    episode_rewards: List[float]  # len = n_envs, each = len(episode)
    episode_lengths: List[int]  # len = n_envs

    vis_observations: Optional[List[ndArray]] = None  # [n_envs, x, y, 3] (3 = #channel)
