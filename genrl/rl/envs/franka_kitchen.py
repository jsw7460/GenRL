import sys
from typing import TypeVar, Any, Tuple, SupportsFloat, Dict, Final

from genrl.rl.envs.utils.skill import GenRLSkillEnv

import numpy as np

sys.path.append("/home/jsw7460/diffusion_rl/")

import gymnasium as gym

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class FrankaKitchenWrapper(GenRLSkillEnv):
    SKILLS: Final[Tuple] = (
        'bottom burner',
        'top burner',
        'light switch',
        'slide cabinet',
        'hinge cabinet',
        'microwave',
        'kettle'
    )

    def __init__(self, env: gym.Env):
        super(FrankaKitchenWrapper, self).__init__(env=env, skills=FrankaKitchenWrapper.SKILLS)
        self.observation_space = gym.spaces.Box(low=-np.infty, high=np.infty, shape=(59,))

        self.subtasks_to_complete = self.env.spec.kwargs["tasks_to_complete"]
        assert len(self.subtasks_to_complete) > 0, "At least one subtask is required to define Franka Kitchen!"

    def reset(
        self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        obs, info = super(FrankaKitchenWrapper, self).reset(seed=seed, options=options)

        obs = obs["observation"]
        return obs, info

    def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, *_ = super(FrankaKitchenWrapper, self).step(action=action)
        obs = obs["observation"]
        return obs, *_

    def __repr__(self):
        subtasks_title = ""
        for subtask in self.subtasks_to_complete:
            subtasks_title += subtask[:2].title()
        return f"Kitchen_{subtasks_title}"
