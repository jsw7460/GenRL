import sys
from functools import partial
from typing import TypeVar, Any, Tuple, SupportsFloat, Dict

import numpy as np

sys.path.append("/home/jsw7460/diffusion_rl/")

import hydra
from omegaconf import DictConfig

from genrl.evaluations.base import EvaluationExecutor
from genrl.evaluations.evaluation_methods import evaluate_policy

import gymnasium as gym

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


class FrankaKitchenWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(FrankaKitchenWrapper, self).__init__(env=env)
        self.observation_space = gym.spaces.Box(low=-np.infty, high=np.infty, shape=(59,))

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


@hydra.main(version_base=None, config_path="../config/eval", config_name="base")
def program(cfg: DictConfig) -> None:
    env = gym.make("FrankaKitchen-v1", max_episode_steps=30)
    env = FrankaKitchenWrapper(env)

    eval_executor = EvaluationExecutor(cfg, envs=[env])
    eval_fn = partial(evaluate_policy, n_eval_episodes=1)
    eval_executor.eval_execute(eval_fn=eval_fn)


if __name__ == "__main__":
    program()
