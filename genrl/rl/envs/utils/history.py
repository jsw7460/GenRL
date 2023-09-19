import collections
from typing import Dict, Tuple, TypeVar, SupportsFloat, Deque, Optional, Union

import gymnasium as gym
import numpy as np

from genrl.utils.common.type_aliases import GenRLEnvOutput, GymEnv

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")
INFTY = 1e+13


class GenRLHistoryEnv(gym.Wrapper):
    """
        Environment wrapper for supporting sequential model inference.
    """

    def __init__(self, env: Union[GymEnv]):
        super(GenRLHistoryEnv, self).__init__(env=env)
        self.env = env
        self.num_stack_frames = None  # Set when setup() method is called.

        self.timestep = 0
        self.goal_stack: Optional[Deque] = None
        self.obs_stack: Optional[Deque] = None
        self.act_stack: Optional[Deque] = None
        self.rew_stack: Optional[Deque] = None
        self.termination_stack: Optional[Deque] = None
        self.truncation_stack: Optional[Deque] = None
        self.info_stack: Optional[Deque] = None

    def setup(self, num_stack_frames: int):
        self.num_stack_frames = num_stack_frames
        self.timestep = 0
        if self.is_goal_conditioned:
            # If env is goal-conditioned, we want to track goal history.
            self.goal_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.obs_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.act_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.rew_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.termination_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.truncation_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.info_stack = collections.deque([], maxlen=self.num_stack_frames)

    def __str__(self):
        return self.env.__str__()

    def __repr__(self):
        return repr(self.env)

    @property
    def observation_space(self):
        """Constructs observation space."""
        parent_obs_space = self.env.observation_space
        act_space = self.action_space
        episode_history = {
            'observations': gym.spaces.Box(
                np.stack([parent_obs_space.low] * self.num_stack_frames, axis=0),
                np.stack([parent_obs_space.high] * self.num_stack_frames, axis=0),
                dtype=parent_obs_space.dtype),
            'actions': gym.spaces.Box(
                np.stack([act_space.low] * self.num_stack_frames, axis=0),
                np.stack([act_space.high] * self.num_stack_frames, axis=0),
                dtype=act_space.dtype
            ),
            'rewards': gym.spaces.Box(-np.inf, np.inf, [self.num_stack_frames])
        }
        if self.is_goal_conditioned:
            goal_shape = np.shape(self.env.goal)  # pytype: disable=attribute-error
            episode_history['returns-to-go'] = gym.spaces.Box(
                -np.inf, np.inf, [self.num_stack_frames] + goal_shape)
        return gym.spaces.Dict(**episode_history)

    @property
    def is_goal_conditioned(self):
        return False

    def pad_current_episode(self, obs, n):
        # Prepad current episode with n steps.
        for _ in range(n):
            if self.is_goal_conditioned:
                self.goal_stack.append(self.env.goal)  # pytype: disable=attribute-error
            self.obs_stack.append(np.zeros_like(obs))
            self.act_stack.append(np.zeros_like(self.env.action_space.sample()))
            self.rew_stack.append(0)
            self.termination_stack.append(0)
            self.truncation_stack.append(0)
            self.info_stack.append(None)

    def _get_observation(self) -> GenRLEnvOutput:
        """Return current episode's N-stacked observation.

        For N=3, the first observation of the episode (reset) looks like:

        *= hasn't happened yet.

        GOAL  OBS  ACT  REW  DONE
        =========================
        g0    0    0.   0.   True
        g0    0    0.   0.   True
        g0    x0   0.   0.   False

        After the first step(a0) taken, yielding x1, r0, done0, info0, the next
        observation looks like:

        GOAL  OBS  ACT  REW  DONE
        =========================
        g0    0    0.   0.   True
        g0    x0   0.   0.   False
        g1    x1   a0   r0   d0

        A more chronologically intuitive way to re-order the column data would be:

        PREV_ACT  PREV_REW  PREV_DONE CURR_GOAL CURR_OBS
        ================================================
        0.        0.        True      g0        0
        0.        0.        False*    g0        x0
        a0        r0        info0     g1        x1

        Returns:
          episode_history: np.ndarray of observation.
        """

        n_mask = min(self.timestep + 1, self.num_stack_frames)
        timesteps = np.arange(self.timestep - self.num_stack_frames + 1, self.timestep + 1)
        timesteps = np.clip(timesteps, a_min=0, a_max=INFTY).astype("i4")

        env_output = GenRLEnvOutput(
            subseq_len=self.num_stack_frames,
            observations=np.stack(self.obs_stack, axis=0),
            actions=np.stack(self.act_stack, axis=0),
            masks=np.concatenate((np.zeros(self.num_stack_frames - n_mask), np.ones(n_mask))),
            timesteps=timesteps,
            rewards=np.stack(self.rew_stack, axis=0),
            terminations=np.stack(self.termination_stack, axis=0),
            truncations=np.stack(self.truncation_stack, axis=0),
            info=self.info_stack
        )

        return env_output

    def reset(self, *args, **kwargs) -> GenRLEnvOutput:
        """Resets env and returns new observation."""
        obs, info = self.env.reset(*args, **kwargs)
        # Create a N-1 "done" past frames.
        self.pad_current_episode(obs, self.num_stack_frames - 1)
        # Create current frame (but with placeholder actions and rewards).
        if self.is_goal_conditioned:
            self.goal_stack.append(self.env.goal)

        self.obs_stack.append(obs)
        self.act_stack.append(np.zeros_like(self.env.action_space.sample()))

        self.rew_stack.append(0)
        self.termination_stack.append(0)
        self.truncation_stack.append(0)
        self.info_stack.append(info)
        return self._get_observation()

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[GenRLEnvOutput, SupportsFloat, bool, bool, Dict]:
        """Replaces env observation with fixed length observation history."""
        # Update applied action to the previous timestep.
        self.timestep += 1
        store_action = action.copy()
        self.act_stack[-1] = store_action
        obs, rew, termination, truncated, info = self.env.step(np.array(action))
        self.rew_stack[-1] = rew

        # Update frame stack.
        self.obs_stack.append(obs)
        self.act_stack.append(
            np.zeros_like(self.env.action_space.sample()))  # Append unknown action to current timestep.
        self.termination_stack.append(termination)
        self.truncation_stack.append(truncated)
        self.info_stack.append(info)

        if self.is_goal_conditioned:
            self.goal_stack.append(self.env.goal)
        if termination:
            if self.is_goal_conditioned:
                # rewrite the observations to reflect hindsight RtG conditioning.
                self.replace_goals_with_hindsight()

        return self._get_observation(), rew, termination, truncated, info
