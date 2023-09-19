import sys
from functools import partial
from typing import TypeVar

sys.path.append("/home/jsw7460/diffusion_rl/")

import hydra
from omegaconf import DictConfig

from vlg.evaluations.base import VLGEvaluationExecutor as EvaluationExecutor
from vlg.evaluations import evaluate_policy
from vlg.rl.envs.franka_kitchen import FrankaKitchenWrapper
from vlg.rl.envs.utils.gym2gymnasium import GymToGymnasium
from vlg.rl.envs.utils.skill_history import VLGHistoryEnv

import gym
import d4rl

_ = d4rl

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


@hydra.main(version_base=None, config_path="../config/eval", config_name="base")
def program(cfg: DictConfig) -> None:
    env = gym.make("kitchen-complete-v0")
    env = GymToGymnasium(env)
    env = FrankaKitchenWrapper(env)
    env = VLGHistoryEnv(env)

    eval_executor = EvaluationExecutor(cfg, envs=(env,))
    eval_fn = partial(evaluate_policy, n_eval_episodes=1)
    eval_executor.eval_execute(eval_fn=eval_fn)


if __name__ == "__main__":
    program()
