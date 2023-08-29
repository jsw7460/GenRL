from typing import Dict, List

import jax

from genrl.utils.common.base import GenRLBaseModule
from genrl.utils.common.base import PolicyNNWrapper
from genrl.utils.common.type_aliases import GenRLPolicyInput, GenRLPolicyOutput
from genrl.utils.interfaces import JaxSavable, Trainable
from genrl.utils.jax_utils.model import Model
from genrl.utils.jax_utils.type_aliases import Params


class BasePolicy(GenRLBaseModule, JaxSavable, Trainable):
    PARAM_NAMES = ["policy_nn"]

    def __init__(self, seed: int, cfg: Dict, init_build_model: bool):
        super(BasePolicy, self).__init__(seed=seed, cfg=cfg, init_build_model=init_build_model)

        self.observation_dim = cfg["observation_dim"]
        self.action_dim = cfg["action_dim"]

        self.optimizer_class = None
        self.policy = None  # type: PolicyNNWrapper
        self.policy_nn: Model

        if init_build_model:
            self.build()

    def build(self):
        pass

    def _excluded_save_params(self) -> List:
        return BasePolicy.PARAM_NAMES

    def _get_save_params(self) -> Dict[str, Params]:
        params_dict = {}
        for str_component in BasePolicy.PARAM_NAMES:
            component = getattr(self, str_component)
            params_dict[str_component] = component.params
        return params_dict

    def _get_load_params(self) -> List[str]:
        return BasePolicy.PARAM_NAMES

    def predict(self, x: GenRLPolicyInput) -> GenRLPolicyOutput:
        return self._predict(x)

    def _predict(self, *args, **kwargs) -> GenRLPolicyOutput:
        raise NotImplementedError()

    def update(self, *args, **kwargs) -> Dict:
        self.n_update += 1
        self.rng, _ = jax.random.split(self.rng)
        return self._update(*args, **kwargs)

    def _update(self, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs) -> Dict:
        pass
