from typing import TypeVar, Tuple

import gymnasium as gym

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class GenRLSkillEnv(gym.Wrapper):
    """
        Environment which outputs a skill information
    """

    def __init__(
        self,
        env: gym.Env,
        skills: Tuple[str, ...]  # Order is important !
    ):
        super(GenRLSkillEnv, self).__init__(env=env)
        self.skills = skills

    def skill2idx(self, skill: str):
        return self.skills.index(skill)

    def idx2skill(self, idx):
        return self.skills[idx]
