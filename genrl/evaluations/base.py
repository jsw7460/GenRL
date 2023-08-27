import pickle
import random
from typing import Dict, Union, Callable

import numpy as np
from hydra.utils import get_class
from omegaconf import DictConfig

from genrl.utils.common.type_aliases import GymEnv
from genrl.utils.interfaces import JaxSavable


class EvaluationExecutor:
    def __init__(self, cfg: Union[Dict, DictConfig], env: GymEnv = None):
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.cfg = cfg
        with open(cfg.pretrained_path, "rb") as f:
            self.pretrained_cfg = pickle.load(f)

        if env is None:
            self.env = self.pretrained_cfg["env_recover"]
        else:
            self.env = env

        self.pretrained_models = {}
        self._load_models()

    def _load_models(self) -> None:
        for module in self.pretrained_cfg["modules"]:
            cls = get_class(self.pretrained_cfg[module]["_target_"])  # type: Union[type, Type[JaxSavable]]
            instance = cls.load(f"{self.pretrained_cfg['save_paths'][module]}_{self.cfg.step}")
            self.pretrained_models[module] = instance

    def eval_execute(self, eval_fn: Callable, **kwargs):
        models = {"model": self.pretrained_models["low_policy"]}
        mean_reward, std_reward = eval_fn(**models, env=self.env, **kwargs)
        print("Run" * 999, mean_reward)
