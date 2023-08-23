from typing import Dict, List

import numpy as np
import optax

from genrl.policies.base import GenRLBaseModule
from genrl.utils.common.type_aliases import GymEnv
from genrl.utils.interfaces import IJaxSavable, ITrainable
from genrl.utils.jax_utils.type_aliases import Params


class BaseLowPolicy(GenRLBaseModule, IJaxSavable, ITrainable):

    def __init__(self, seed: int, env: GymEnv, cfg: Dict, init_build_model: bool):
        super(BaseLowPolicy, self).__init__(sed=seed, env=env, cfg=cfg, init_build_model=init_build_model)
        self.optimizer_class = None
        self.policy = None

        if init_build_model:
            self.build()

    def build(self):
        optimizer = getattr(optax, self.cfg["optimizer_class"])
        self.optimizer_class = optimizer

    def _excluded_save_params(self) -> List:
        pass

    def _get_save_params(self) -> Dict[str, Params]:
        pass

    def _get_load_params(self) -> List[str]:
        pass

    def build(self):
        pass

    def predict(self, *args, **kwargs) -> np.ndarray:
        pass

    def update(self, *args, **kwargs) -> Dict:
        pass

    def evaluate(self, *args, **kwargs) -> Dict:
        pass
