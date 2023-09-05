import sys
from functools import partial
from typing import TypeVar

sys.path.append("/home/jsw7460/diffusion_rl/")

import hydra
from omegaconf import DictConfig

from genrl.rl.envs.franka_kitchen import FrankaKitchenWrapper
from genrl.rl.envs.utils import GenRLHistoryEnv

from genrl.evaluations.base import EvaluationExecutor
from genrl.evaluations.evaluation_methods import evaluate_policy

import gymnasium as gym

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


@hydra.main(version_base=None, config_path="../config/eval", config_name="base")
def program(cfg: DictConfig) -> None:
    env1 = gym.make(
        "FrankaKitchen-v1",
        tasks_to_complete=["microwave", "kettle", "light switch"],
        max_episode_steps=280,
        render_mode="rgb_array"
    )
    env1 = GenRLHistoryEnv(FrankaKitchenWrapper(env1))

    env2 = gym.make(
        "FrankaKitchen-v1",
        tasks_to_complete=["kettle", "light switch", "microwave"],
        max_episode_steps=280,
        render_mode="rgb_array"
    )
    env2 = GenRLHistoryEnv(FrankaKitchenWrapper(env2))

    eval_executor = EvaluationExecutor(cfg, envs=(env1, env2))
    eval_fn = partial(evaluate_policy, n_eval_episodes=1)
    eval_executor.eval_execute(eval_fn=eval_fn)


if __name__ == "__main__":
    program()
