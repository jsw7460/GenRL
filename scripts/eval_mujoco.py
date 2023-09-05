import sys
from functools import partial
from typing import TypeVar

sys.path.append("/home/jsw7460/diffusion_rl/")

from genrl.rl.envs.utils import GenRLHistoryEnv

import hydra
from omegaconf import DictConfig

from genrl.evaluations.base import EvaluationExecutor
from genrl.evaluations.evaluation_methods import evaluate_policy

import gymnasium as gym

WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")


@hydra.main(version_base=None, config_path="../config/eval", config_name="base")
def program(cfg: DictConfig) -> None:
    env1 = GenRLHistoryEnv(gym.make("HalfCheetah-v4"))

    eval_executor = EvaluationExecutor(cfg, envs=(env1,))
    eval_fn = partial(evaluate_policy, n_eval_episodes=1)
    eval_executor.eval_execute(eval_fn=eval_fn)


if __name__ == "__main__":
    program()
